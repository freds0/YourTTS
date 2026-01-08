"""PyTorch Lightning implementation of YourTTS/VITS model.

This module provides a PyTorch Lightning wrapper around the VITS model,
enabling modern training workflows with features like:
- Automatic device placement
- Distributed training
- Gradient clipping
- Learning rate scheduling
- TensorBoard logging
- Checkpointing

The module maintains backward compatibility with existing YourTTS checkpoints.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
from typing import Dict, List, Optional, Tuple, Any
from torch.cuda.amp.autocast_mode import autocast

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, wav_to_mel
from TTS.tts.layers.losses import VitsGeneratorLoss, VitsDiscriminatorLoss
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.helpers import segment
from TTS.vocoder.utils.generic_utils import plot_results
from TTS.tts.utils.visual import plot_alignment


class YourTTSLightningModule(pl.LightningModule):
    """PyTorch Lightning module for YourTTS/VITS training.

    This module wraps the VITS model and implements all necessary PyTorch Lightning
    hooks for training, validation, and optimization. It uses a dual-optimizer
    approach (discriminator + generator) typical of GAN-based models.

    Args:
        config (VitsConfig): Model configuration
        ap (AudioProcessor, optional): Audio processor for mel-spectrogram conversion
        tokenizer (TTSTokenizer, optional): Text tokenizer
        speaker_manager (SpeakerManager, optional): Manager for speaker embeddings
        language_manager (LanguageManager, optional): Manager for language embeddings

    Example:
        >>> config = VitsConfig()
        >>> model = YourTTSLightningModule(config)
        >>> trainer = pl.Trainer(max_epochs=1000, gpus=1)
        >>> trainer.fit(model, train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        config: VitsConfig,
        ap: Optional["AudioProcessor"] = None,
        tokenizer: Optional[TTSTokenizer] = None,
        speaker_manager: Optional[SpeakerManager] = None,
        language_manager: Optional[LanguageManager] = None,
    ):
        super().__init__()

        # Store config and hyperparameters
        self.config = config
        self.save_hyperparameters(ignore=['ap', 'tokenizer', 'speaker_manager', 'language_manager'])

        # Initialize the VITS model (contains all neural network components)
        self.vits_model = Vits(
            config=config,
            ap=ap,
            tokenizer=tokenizer,
            speaker_manager=speaker_manager,
            language_manager=language_manager,
        )

        # Store references for easy access
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager

        # Initialize loss functions
        self.generator_loss = VitsGeneratorLoss(self.config)
        self.discriminator_loss = VitsDiscriminatorLoss(self.config)

        # Cache for outputs between discriminator and generator steps
        self.model_outputs_cache = None

        # Automatic optimization is disabled because we use 2 optimizers
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        """Forward pass through the VITS model."""
        return self.vits_model.forward(*args, **kwargs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Training step for both discriminator and generator.

        This method handles the alternating training of discriminator and generator.
        PyTorch Lightning will call this twice per batch (once for each optimizer).

        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and logging information
        """
        # Get both optimizers
        opt_disc, opt_gen = self.optimizers()

        # Extract batch data
        tokens = batch["token_id"]
        token_lens = batch["token_id_lengths"]
        # Transpose from [B, T, D] to [B, D, T] for VITS model
        spec = batch["linear"].transpose(1, 2)
        spec_lens = batch["mel_lengths"]
        # Transpose waveform from [B, T, 1] to [B, 1, T]
        waveform = batch["waveform"].transpose(1, 2)
        # Transpose mel from [B, T, D] to [B, D, T]
        mel = batch["mel"].transpose(1, 2)
        d_vectors = batch.get("d_vectors")
        speaker_ids = batch.get("speaker_ids")
        language_ids = batch.get("language_ids")

        # ========================================
        # DISCRIMINATOR TRAINING STEP
        # ========================================
        opt_disc.zero_grad()

        # Generate fake samples
        with torch.no_grad():
            outputs = self.vits_model.forward(
                tokens,
                token_lens,
                spec,
                spec_lens,
                waveform,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

        # Compute discriminator scores
        scores_disc_fake, _, scores_disc_real, _ = self.vits_model.disc(
            outputs["model_outputs"].detach(), outputs["waveform_seg"]
        )

        # Compute discriminator loss
        with autocast(enabled=False):
            loss_dict_disc = self.discriminator_loss(
                scores_disc_real,
                scores_disc_fake,
            )

        loss_disc = loss_dict_disc["loss"]

        # Backward pass for discriminator
        self.manual_backward(loss_disc)

        # Gradient clipping (grad_clip is a list: [gen, disc])
        grad_clip_disc = self.config.grad_clip[1] if isinstance(self.config.grad_clip, list) else self.config.grad_clip
        if grad_clip_disc > 0:
            torch.nn.utils.clip_grad_norm_(
                self.vits_model.disc.parameters(),
                grad_clip_disc
            )

        # Update discriminator
        opt_disc.step()

        # ========================================
        # GENERATOR TRAINING STEP
        # ========================================
        opt_gen.zero_grad()

        # Full forward pass
        outputs = self.vits_model.forward(
            tokens,
            token_lens,
            spec,
            spec_lens,
            waveform,
            aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
        )

        # Compute mel spectrogram segment for generator loss
        with autocast(enabled=False):
            if self.vits_model.args.encoder_sample_rate:
                spec_segment_size = self.vits_model.spec_segment_size * int(self.vits_model.interpolate_factor)
            else:
                spec_segment_size = self.vits_model.spec_segment_size

            mel_slice = segment(
                mel.float(), outputs["slice_ids"], spec_segment_size, pad_short=True
            )
            mel_slice_hat = wav_to_mel(
                y=outputs["model_outputs"].float(),
                n_fft=self.config.audio.fft_size,
                sample_rate=self.config.audio.sample_rate,
                num_mels=self.config.audio.num_mels,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
                center=False,
            )

        # Compute discriminator scores and features for generator loss
        scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.vits_model.disc(
            outputs["model_outputs"], outputs["waveform_seg"]
        )

        # Compute generator loss
        with autocast(enabled=False):
            loss_dict_gen = self.generator_loss(
                mel_slice_hat=mel_slice_hat.float(),
                mel_slice=mel_slice.float(),
                z_p=outputs["z_p"].float(),
                logs_q=outputs["logs_q"].float(),
                m_p=outputs["m_p"].float(),
                logs_p=outputs["logs_p"].float(),
                z_len=spec_lens,
                scores_disc_fake=scores_disc_fake,
                feats_disc_fake=feats_disc_fake,
                feats_disc_real=feats_disc_real,
                loss_duration=outputs["loss_duration"],
                use_speaker_encoder_as_loss=self.vits_model.args.use_speaker_encoder_as_loss,
                gt_spk_emb=outputs.get("gt_spk_emb"),
                syn_spk_emb=outputs.get("syn_spk_emb"),
            )

        loss_gen = loss_dict_gen["loss"]

        # Backward pass for generator
        self.manual_backward(loss_gen)

        # Gradient clipping (grad_clip is a list: [gen, disc])
        grad_clip_gen = self.config.grad_clip[0] if isinstance(self.config.grad_clip, list) else self.config.grad_clip
        if grad_clip_gen > 0:
            torch.nn.utils.clip_grad_norm_(
                chain(
                    self.vits_model.text_encoder.parameters(),
                    self.vits_model.posterior_encoder.parameters(),
                    self.vits_model.flow.parameters(),
                    self.vits_model.waveform_decoder.parameters(),
                    self.vits_model.duration_predictor.parameters(),
                ),
                grad_clip_gen
            )

        # Update generator
        opt_gen.step()

        # ========================================
        # LOGGING
        # ========================================
        # Log discriminator losses
        for key, value in loss_dict_disc.items():
            self.log(f"train/disc_{key}", value, prog_bar=False, logger=True)

        # Log generator losses
        for key, value in loss_dict_gen.items():
            self.log(f"train/gen_{key}", value, prog_bar=(key == "loss"), logger=True)

        # Log combined loss for progress bar
        total_loss = loss_disc + loss_gen
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True)

        # Store outputs for potential logging callbacks
        self.model_outputs_cache = outputs

        return {"loss": total_loss, "outputs": outputs}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step (generator only, no discriminator).

        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing validation loss
        """
        # Extract batch data
        tokens = batch["token_id"]
        token_lens = batch["token_id_lengths"]

        # Transpose from [B, T, D] to [B, D, T] for VITS model
        spec = batch["linear"].transpose(1, 2)
        spec_lens = batch["mel_lengths"]

        # Transpose waveform from [B, T, 1] to [B, 1, T]
        waveform = batch["waveform"].transpose(1, 2)
        # Transpose mel from [B, T, D] to [B, D, T]
        mel = batch["mel"].transpose(1, 2)

        d_vectors = batch.get("d_vectors")
        speaker_ids = batch.get("speaker_ids")
        language_ids = batch.get("language_ids")

        # Forward pass
        outputs = self.vits_model.forward(
            tokens,
            token_lens,
            spec,
            spec_lens,
            waveform,
            aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
        )

        # Compute mel spectrogram segment
        with autocast(enabled=False):
            if self.vits_model.args.encoder_sample_rate:
                spec_segment_size = self.vits_model.spec_segment_size * int(self.vits_model.interpolate_factor)
            else:
                spec_segment_size = self.vits_model.spec_segment_size

            mel_slice = segment(
                mel.float(), outputs["slice_ids"], spec_segment_size, pad_short=True
            )
            mel_slice_hat = wav_to_mel(
                y=outputs["model_outputs"].float(),
                n_fft=self.config.audio.fft_size,
                sample_rate=self.config.audio.sample_rate,
                num_mels=self.config.audio.num_mels,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
                center=False,
            )

        # Compute discriminator scores and features (for feature matching loss)
        scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.vits_model.disc(
            outputs["model_outputs"], outputs["waveform_seg"]
        )

        # Compute validation loss
        with autocast(enabled=False):
            loss_dict = self.generator_loss(
                mel_slice_hat=mel_slice_hat.float(),
                mel_slice=mel_slice.float(),
                z_p=outputs["z_p"].float(),
                logs_q=outputs["logs_q"].float(),
                m_p=outputs["m_p"].float(),
                logs_p=outputs["logs_p"].float(),
                z_len=spec_lens,
                scores_disc_fake=scores_disc_fake,
                feats_disc_fake=feats_disc_fake,
                feats_disc_real=feats_disc_real,
                loss_duration=outputs["loss_duration"],
                use_speaker_encoder_as_loss=self.vits_model.args.use_speaker_encoder_as_loss,
                gt_spk_emb=outputs.get("gt_spk_emb"),
                syn_spk_emb=outputs.get("syn_spk_emb"),
            )

        # Log validation losses
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, prog_bar=(key == "loss"), logger=True, sync_dist=True)

        return {"val_loss": loss_dict["loss"], "outputs": outputs}

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns two optimizers:
        1. Discriminator optimizer
        2. Generator optimizer (text_encoder, posterior_encoder, flow, decoder, duration_predictor)
        """
        # Discriminator optimizer
        opt_disc = torch.optim.AdamW(
            self.vits_model.disc.parameters(),
            lr=self.config.lr_disc,
            #betas=[self.config.beta1_disc, self.config.beta2_disc],
            #weight_decay=self.config.wd_disc,
        )

        # Generator optimizer
        opt_gen = torch.optim.AdamW(
            chain(
                self.vits_model.text_encoder.parameters(),
                self.vits_model.posterior_encoder.parameters(),
                self.vits_model.flow.parameters(),
                self.vits_model.waveform_decoder.parameters(),
                self.vits_model.duration_predictor.parameters(),
            ),
            lr=self.config.lr_gen,
            #betas=[self.config.beta1_gen, self.config.beta2_gen],
            #weight_decay=self.config.wd_gen,
        )

        # Learning rate schedulers
        schedulers = []

        if self.config.lr_scheduler_gen == "ExponentialLR":
            scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(
                opt_gen,
                gamma=self.config.lr_scheduler_gen_params.get("gamma", 0.999875)
            )
            schedulers.append({
                "scheduler": scheduler_gen,
                "interval": "step",
                "frequency": 1,
            })

        if self.config.lr_scheduler_disc == "ExponentialLR":
            scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(
                opt_disc,
                gamma=self.config.lr_scheduler_disc_params.get("gamma", 0.999875)
            )
            schedulers.append({
                "scheduler": scheduler_disc,
                "interval": "step",
                "frequency": 1,
            })

        if schedulers:
            return [opt_disc, opt_gen], schedulers
        else:
            return [opt_disc, opt_gen]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint.

        This method ensures compatibility with old checkpoint format.
        We save both Lightning format and legacy format.
        """
        # Add legacy format for compatibility
        checkpoint["model"] = self.vits_model.state_dict()
        checkpoint["config"] = self.config

        # Add speaker and language managers if present
        if self.speaker_manager is not None:
            checkpoint["speaker_manager"] = self.speaker_manager
        if self.language_manager is not None:
            checkpoint["language_manager"] = self.language_manager

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dict with backward compatibility.

        Supports loading:
        1. New PyTorch Lightning checkpoints (state_dict key)
        2. Old Trainer checkpoints (model key with optimizer, scaler, etc.)
        3. Legacy checkpoints from old training framework
        """
        # Check if this is an old checkpoint format (has 'model' key but not 'state_dict')
        if "model" in state_dict and "state_dict" not in state_dict:
            # Old format - load VITS model weights directly
            print("Loading from old checkpoint format (model key)")
            self.vits_model.load_state_dict(state_dict["model"], strict=strict)
        else:
            # New Lightning format
            super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[Any] = None,
        **kwargs
    ) -> "YourTTSLightningModule":
        """Load model from checkpoint with backward compatibility.

        Supports loading from:
        1. PyTorch Lightning checkpoints (new format)
        2. Old Trainer checkpoints with 'model', 'optimizer', 'scaler', etc.
        3. Legacy checkpoints from the old training framework

        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to load checkpoint to
            **kwargs: Additional arguments to pass to __init__
                - config: Model config (optional, will be extracted from checkpoint if not provided)
                - ap: AudioProcessor instance
                - tokenizer: TTSTokenizer instance
                - speaker_manager: SpeakerManager instance
                - language_manager: LanguageManager instance

        Returns:
            Loaded YourTTSLightningModule instance

        Example:
            >>> # Load from old checkpoint
            >>> model = YourTTSLightningModule.load_from_checkpoint(
            ...     "checkpoint_1020000.pth",
            ...     config=config,  # Optional if checkpoint has config
            ...     ap=ap,
            ...     tokenizer=tokenizer,
            ...     speaker_manager=speaker_manager,
            ... )
        """
        # Load checkpoint (weights_only=False needed for custom config objects)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Detect checkpoint format
        is_old_format = "model" in checkpoint and "state_dict" not in checkpoint
        is_lightning_format = "state_dict" in checkpoint

        print(f"Loading checkpoint from: {checkpoint_path}")
        if is_old_format:
            print("Detected old checkpoint format (contains 'model', 'optimizer', 'scaler', etc.)")
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        elif is_lightning_format:
            print("Detected PyTorch Lightning checkpoint format")
        else:
            print(f"Warning: Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

        # Extract config - prioritize kwargs if provided
        if "config" in kwargs:
            config = kwargs.pop("config")
            print("Using config from kwargs")
        elif "hyper_parameters" in checkpoint:
            # Lightning checkpoint
            config = checkpoint["hyper_parameters"].get("config")
            print("Extracted config from Lightning checkpoint hyper_parameters")
        elif "config" in checkpoint:
            # Old checkpoint
            config = checkpoint["config"]
            print("Extracted config from checkpoint")
        else:
            raise ValueError(
                "Cannot find config in checkpoint. "
                "Please provide config via kwargs: load_from_checkpoint(..., config=your_config)"
            )

        # Create model instance
        print("Creating model instance...")
        model = cls(config=config, **kwargs)

        # Load weights based on checkpoint format
        if is_old_format:
            print("Loading model weights from 'model' key...")
            # Old format: checkpoint["model"] contains the VITS model state_dict

            # Load with strict=False to handle size mismatches gracefully
            try:
                missing_keys, unexpected_keys = model.vits_model.load_state_dict(
                    checkpoint["model"], strict=False
                )

                if missing_keys:
                    print(f"\nWarning: Missing keys in checkpoint ({len(missing_keys)} keys)")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            print(f"  - {key}")
                    else:
                        for key in missing_keys[:5]:
                            print(f"  - {key}")
                        print(f"  ... and {len(missing_keys) - 5} more")
                    print("These parameters will be randomly initialized.\n")

                if unexpected_keys:
                    print(f"\nWarning: Unexpected keys in checkpoint ({len(unexpected_keys)} keys)")
                    if len(unexpected_keys) <= 10:
                        for key in unexpected_keys:
                            print(f"  - {key}")
                    else:
                        for key in unexpected_keys[:5]:
                            print(f"  - {key}")
                        print(f"  ... and {len(unexpected_keys) - 5} more")
                    print("These parameters will be ignored.\n")

                print("Successfully loaded model weights from old checkpoint")

            except RuntimeError as e:
                error_msg = str(e)
                if "size mismatch" in error_msg:
                    print("\n⚠️  Size mismatch detected! Attempting to load with size adaptation...")
                    print(f"Error details: {error_msg}\n")

                    # Load weights manually, skipping mismatched sizes
                    model_state = model.vits_model.state_dict()
                    checkpoint_state = checkpoint["model"]

                    loaded_keys = []
                    skipped_keys = []

                    for key, checkpoint_param in checkpoint_state.items():
                        if key in model_state:
                            model_param = model_state[key]
                            if checkpoint_param.shape == model_param.shape:
                                model_state[key] = checkpoint_param
                                loaded_keys.append(key)
                            else:
                                skipped_keys.append((key, checkpoint_param.shape, model_param.shape))
                        else:
                            skipped_keys.append((key, checkpoint_param.shape, "not in model"))

                    # Load the adapted state dict
                    model.vits_model.load_state_dict(model_state)

                    print(f"✓ Loaded {len(loaded_keys)} parameters successfully")
                    print(f"⚠️  Skipped {len(skipped_keys)} parameters due to shape mismatch:\n")

                    for key, ckpt_shape, model_shape in skipped_keys[:10]:
                        print(f"  - {key}")
                        print(f"    Checkpoint shape: {ckpt_shape}")
                        print(f"    Model shape: {model_shape}")

                    if len(skipped_keys) > 10:
                        print(f"  ... and {len(skipped_keys) - 10} more")

                    print("\nParameters with shape mismatch will be randomly initialized.")
                    print("This is common when vocabulary size changes or model architecture is updated.\n")
                else:
                    # Re-raise if it's a different error
                    raise

            # Optionally print metadata if available
            if "epoch" in checkpoint:
                print(f"Checkpoint metadata - Epoch: {checkpoint['epoch']}, Step: {checkpoint.get('step', 'N/A')}")
            if "model_loss" in checkpoint:
                print(f"Checkpoint model loss: {checkpoint['model_loss']}")
        else:
            # Lightning format or unknown format
            print("Loading weights using Lightning's load_state_dict...")
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded model weights")

        return model

    def inference(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        d_vector: Optional[torch.Tensor] = None,
        language_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run inference to generate speech from text.

        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID (for multi-speaker models with embedding)
            d_vector: Speaker d-vector (for models using external embeddings)
            language_id: Language ID (for multilingual models)
            length_scale: Duration scale (higher = slower speech)
            noise_scale: Noise scale for variation

        Returns:
            Generated waveform tensor
        """
        self.eval()
        with torch.no_grad():
            # Tokenize text
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")

            tokens = self.tokenizer.text_to_ids(text)
            tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
            token_lens = torch.LongTensor([len(tokens[0])]).to(self.device)

            # Prepare auxiliary inputs
            aux_input = {}
            if speaker_id is not None:
                aux_input["speaker_ids"] = torch.LongTensor([speaker_id]).to(self.device)
            if d_vector is not None:
                aux_input["d_vectors"] = d_vector.unsqueeze(0).to(self.device)
            if language_id is not None:
                aux_input["language_ids"] = torch.LongTensor([language_id]).to(self.device)

            # Inference
            outputs = self.vits_model.inference(
                tokens,
                aux_input=aux_input,
                d_vector=d_vector,
                speaker_id=speaker_id,
                language_id=language_id,
            )

            wav = outputs["model_outputs"]

        return wav
