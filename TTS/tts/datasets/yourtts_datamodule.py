"""PyTorch Lightning DataModule for YourTTS training.

This module handles all data loading, preprocessing, and batching for YourTTS training.
It supports:
- Multi-speaker datasets with d-vectors
- Multilingual datasets
- Automatic speaker embedding computation
- Efficient bucketing for variable-length sequences
"""

from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer.torch import DistributedSampler, DistributedSamplerWrapper


class YourTTSDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for YourTTS.

    This class handles all aspects of data preparation including:
    - Loading training and validation samples
    - Computing speaker embeddings (d-vectors) if needed
    - Creating dataloaders with appropriate samplers
    - Handling distributed training

    Args:
        config (VitsConfig): Model configuration
        train_samples (List[Dict]): List of training samples
        eval_samples (List[Dict]): List of evaluation samples
        tokenizer (TTSTokenizer): Text tokenizer
        speaker_manager (SpeakerManager, optional): Manager for speaker embeddings
        language_manager (LanguageManager, optional): Manager for language embeddings
        ap (AudioProcessor, optional): Audio processor

    Example:
        >>> config = VitsConfig()
        >>> datamodule = YourTTSDataModule(config, train_samples, eval_samples, tokenizer)
        >>> trainer = pl.Trainer()
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        config: VitsConfig,
        train_samples: List[Dict[str, Any]],
        eval_samples: List[Dict[str, Any]],
        tokenizer: TTSTokenizer,
        speaker_manager: Optional[SpeakerManager] = None,
        language_manager: Optional[LanguageManager] = None,
        ap: Optional[AudioProcessor] = None,
    ):
        super().__init__()
        self.config = config
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self.ap = ap

        # Datasets will be created in setup()
        self.train_dataset = None
        self.eval_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training or validation.

        This method is called by PyTorch Lightning before training/validation starts.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            # Create training dataset
            self.train_dataset = TTSDataset(
                outputs_per_step=self.config.r if hasattr(self.config, 'r') else 1,
                compute_linear_spec=True,
                samples=self.train_samples,
                ap=self.ap,
                return_wav=True,
                batch_group_size=0,  # Handled by BucketBatchSampler
                min_text_len=self.config.min_text_len if hasattr(self.config, 'min_text_len') else 1,
                max_text_len=self.config.max_text_len if hasattr(self.config, 'max_text_len') else float("inf"),
                min_audio_len=self.config.min_audio_len if hasattr(self.config, 'min_audio_len') else 1,
                max_audio_len=self.config.max_audio_len if hasattr(self.config, 'max_audio_len') else float("inf"),
                phoneme_cache_path=self.config.phoneme_cache_path if hasattr(self.config, 'phoneme_cache_path') else None,
                precompute_num_workers=self.config.precompute_num_workers if hasattr(self.config, 'precompute_num_workers') else 0,
                verbose=False,
                tokenizer=self.tokenizer,
                speaker_id_mapping=self.speaker_manager.name_to_id if self.speaker_manager else None,
                d_vector_mapping=self.speaker_manager.embeddings if self.speaker_manager and hasattr(self.speaker_manager, 'embeddings') else None,
                language_id_mapping=self.language_manager.language_id_mapping if self.language_manager else None,
                use_noise_augment=self.config.use_noise_augment if hasattr(self.config, 'use_noise_augment') else False,
                start_by_longest=True,
            )

            # Create validation dataset
            self.eval_dataset = TTSDataset(
                outputs_per_step=self.config.r if hasattr(self.config, 'r') else 1,
                compute_linear_spec=True,
                samples=self.eval_samples,
                ap=self.ap,
                return_wav=True,
                batch_group_size=0,
                min_text_len=self.config.min_text_len if hasattr(self.config, 'min_text_len') else 1,
                max_text_len=self.config.max_text_len if hasattr(self.config, 'max_text_len') else float("inf"),
                min_audio_len=self.config.min_audio_len if hasattr(self.config, 'min_audio_len') else 1,
                max_audio_len=self.config.max_audio_len if hasattr(self.config, 'max_audio_len') else float("inf"),
                phoneme_cache_path=self.config.phoneme_cache_path if hasattr(self.config, 'phoneme_cache_path') else None,
                precompute_num_workers=0,
                verbose=False,
                tokenizer=self.tokenizer,
                speaker_id_mapping=self.speaker_manager.name_to_id if self.speaker_manager else None,
                d_vector_mapping=self.speaker_manager.embeddings if self.speaker_manager and hasattr(self.speaker_manager, 'embeddings') else None,
                language_id_mapping=self.language_manager.language_id_mapping if self.language_manager else None,
                use_noise_augment=False,  # No augment for eval
                start_by_longest=True,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader for training
        """
        # Handle weighted sampling if configured
        sampler = None
        if hasattr(self.config, 'use_weighted_sampler') and self.config.use_weighted_sampler:
            # Create weights based on language/speaker distribution
            weights = self._compute_sample_weights()
            sampler = WeightedRandomSampler(weights, len(weights))

            if self.trainer and self.trainer.world_size > 1:
                sampler = DistributedSamplerWrapper(sampler)
        elif self.trainer and self.trainer.world_size > 1:
            # Use distributed sampler for multi-GPU
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.config.num_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader for validation
        """
        # For validation, we use simpler sampling without bucketing
        sampler = None
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )

        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size if hasattr(self.config, 'eval_batch_size') else self.config.batch_size,
            sampler=sampler,
            collate_fn=self.eval_dataset.collate_fn,
            num_workers=self.config.num_loader_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def _compute_sample_weights(self) -> List[float]:
        """Compute sample weights for weighted sampling.

        This helps balance the dataset when you have:
        - Imbalanced speaker distribution
        - Imbalanced language distribution

        Returns:
            List of weights, one per sample
        """
        weights = []

        # Count samples per speaker/language
        if self.language_manager:
            # Multilingual: weight by language
            language_counts = {}
            for sample in self.train_samples:
                lang = sample.get("language", "default")
                language_counts[lang] = language_counts.get(lang, 0) + 1

            # Inverse frequency weighting
            total_samples = len(self.train_samples)
            for sample in self.train_samples:
                lang = sample.get("language", "default")
                weight = total_samples / (len(language_counts) * language_counts[lang])
                weights.append(weight)

        elif self.speaker_manager and self.speaker_manager.num_speakers > 1:
            # Multi-speaker: weight by speaker
            speaker_counts = {}
            for sample in self.train_samples:
                speaker = sample.get("speaker_name", "default")
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            # Inverse frequency weighting
            total_samples = len(self.train_samples)
            for sample in self.train_samples:
                speaker = sample.get("speaker_name", "default")
                weight = total_samples / (len(speaker_counts) * speaker_counts[speaker])
                weights.append(weight)

        else:
            # Equal weights
            weights = [1.0] * len(self.train_samples)

        return weights

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Hook called before batch is transferred to device.

        This can be used for any final preprocessing.

        Args:
            batch: The batch data
            dataloader_idx: Index of the dataloader

        Returns:
            Processed batch
        """
        return batch

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/validation.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Clean up datasets to free memory
        if stage == "fit":
            self.train_dataset = None
            self.eval_dataset = None


def prepare_yourtts_data(
    config: VitsConfig,
    dataset_config: Dict[str, Any],
    output_path: str,
    speaker_manager: Optional[SpeakerManager] = None,
) -> tuple:
    """Prepare training and validation samples for YourTTS.

    This is a helper function to load and prepare data before creating the DataModule.

    Args:
        config: Model configuration
        dataset_config: Dataset configuration with paths and metadata
        output_path: Path to save computed embeddings
        speaker_manager: Speaker manager for computing d-vectors

    Returns:
        Tuple of (train_samples, eval_samples, tokenizer, speaker_manager, language_manager, ap)

    Example:
        >>> train_samples, eval_samples, tokenizer, speaker_manager, lang_manager, ap = prepare_yourtts_data(
        ...     config, dataset_config, output_path
        ... )
        >>> datamodule = YourTTSDataModule(config, train_samples, eval_samples, tokenizer, speaker_manager, lang_manager, ap)
    """
    from TTS.tts.datasets import load_tts_samples
    from TTS.utils.audio import AudioProcessor

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize tokenizer
    tokenizer, _ = TTSTokenizer.init_from_config(config)

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size if hasattr(config, 'eval_split_max_size') else None,
        eval_split_size=config.eval_split_size if hasattr(config, 'eval_split_size') else 0.01,
    )

    # Initialize language manager if multilingual
    language_manager = None
    if config.model_args.use_language_embedding:
        language_manager = LanguageManager(config=config)

    # Compute speaker embeddings (d-vectors) if using external embeddings
    if config.model_args.use_d_vector_file and speaker_manager:
        from TTS.bin.compute_embeddings import compute_embeddings

        # Compute embeddings for all samples
        compute_embeddings(
            speaker_manager.encoder,
            speaker_manager.encoder_ap,
            train_samples + eval_samples,
            output_path,
        )

        # Update speaker manager with computed embeddings
        speaker_manager.load_embeddings_from_list(train_samples + eval_samples)

    return train_samples, eval_samples, tokenizer, speaker_manager, language_manager, ap
