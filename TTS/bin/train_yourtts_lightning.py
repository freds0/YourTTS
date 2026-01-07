#!/usr/bin/env python3
"""Training script for YourTTS using PyTorch Lightning.

This script provides a modern training workflow for YourTTS with:
- Multi-GPU training with DDP
- Automatic mixed precision
- Gradient clipping
- Learning rate scheduling
- TensorBoard logging
- Model checkpointing
- Resume from checkpoint

Usage:
    # Single GPU training
    python TTS/bin/train_yourtts_lightning.py --config_path configs/yourtts_config.json

    # Multi-GPU training
    python TTS/bin/train_yourtts_lightning.py --config_path configs/yourtts_config.json --gpus 4

    # Resume from checkpoint
    python TTS/bin/train_yourtts_lightning.py --config_path configs/yourtts_config.json --resume_from_checkpoint path/to/checkpoint.ckpt

Example config structure:
    {
        "model": "vits",
        "batch_size": 32,
        "eval_batch_size": 16,
        "num_loader_workers": 8,
        "num_eval_loader_workers": 4,
        "run_eval": true,
        "test_delay_epochs": 10,
        "epochs": 1000,
        "text_cleaner": "phoneme_cleaners",
        "use_phonemes": true,
        "phoneme_language": "en-us",
        "print_step": 25,
        "print_eval": true,
        "mixed_precision": true,
        "output_path": "outputs/yourtts_experiment/",
        ...
    }
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.datasets.yourtts_datamodule import YourTTSDataModule, prepare_yourtts_data
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
#from TTS.utils.generic_utils import setup_model_from_config
#from TTS.utils.audio import AudioProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YourTTS model using PyTorch Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to config JSON file"
    )

    parser.add_argument(
        "--restore_path",
        type=str,
        default=None,
        help="Path to model checkpoint to restore from (for fine-tuning or continuing training)"
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator to use (default: gpu)"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 16-mixed for mixed precision)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Training strategy (auto, ddp, ddp_spawn, deepspeed, etc.)"
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to Lightning checkpoint to resume training from"
    )

    parser.add_argument(
        "--skip_train_epoch",
        action="store_true",
        help="Skip training and only run evaluation (useful for testing)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fast_dev_run with 5 batches)"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="yourtts-training",
        help="W&B project name (default: yourtts-training)"
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team name)"
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )

    return parser.parse_args()


def setup_speaker_manager(config: VitsConfig, samples: list) -> SpeakerManager:
    """Setup speaker manager with d-vectors if needed.

    Args:
        config: Model configuration
        samples: List of dataset samples

    Returns:
        Initialized SpeakerManager
    """
    speaker_manager = None

    if config.model_args.use_speaker_embedding or config.model_args.use_d_vector_file:
        speaker_manager = SpeakerManager()

        # Get unique speakers from samples
        speaker_manager.set_ids_from_data(samples, parse_key="speaker_name")

        # Load speaker encoder if using d-vectors
        if config.model_args.use_d_vector_file:
            if config.model_args.use_speaker_encoder_as_loss:
                # Load speaker encoder for both d-vectors and SCL loss
                speaker_manager.init_encoder(
                    config.model_args.speaker_encoder_model_path,
                    config.model_args.speaker_encoder_config_path,
                    use_cuda=torch.cuda.is_available(),
                )

            # Load precomputed d-vectors if available
            if config.model_args.d_vector_file and os.path.exists(config.model_args.d_vector_file[0]):
                speaker_manager.load_embeddings_from_file(config.model_args.d_vector_file[0])

    return speaker_manager


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = VitsConfig()
    config.load_json(args.config_path)

    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)

    print("\n" + "="*80)
    print("YourTTS Training with PyTorch Lightning")
    print("="*80)
    print(f"Config: {args.config_path}")
    print(f"Output: {config.output_path}")
    print(f"GPUs: {args.gpus}")
    print(f"Precision: {args.precision}")
    print("="*80 + "\n")

    # ========================================
    # Prepare Data
    # ========================================
    print("📊 Preparing data...")

    # Load dataset config
    # This assumes your config has dataset information
    dataset_config = config.datasets if hasattr(config, 'datasets') else []
    # Prepare samples
    train_samples, eval_samples, tokenizer, speaker_manager, language_manager, ap = prepare_yourtts_data(
        config=config,
        dataset_config=dataset_config,
        output_path=config.output_path,
        speaker_manager=None,  # Will be created inside if needed
    )

    print(f"✓ Loaded {len(train_samples)} training samples")
    print(f"✓ Loaded {len(eval_samples)} validation samples")

    if speaker_manager:
        print(f"✓ Speakers: {speaker_manager.num_speakers}")
    if language_manager:
        print(f"✓ Languages: {language_manager.num_languages}")

    # ========================================
    # Create DataModule
    # ========================================
    print("\n📦 Creating data module...")

    datamodule = YourTTSDataModule(
        config=config,
        train_samples=train_samples,
        eval_samples=eval_samples,
        tokenizer=tokenizer,
        speaker_manager=speaker_manager,
        language_manager=language_manager,
        ap=ap,
    )

    print("✓ DataModule created")

    # ========================================
    # Create Model
    # ========================================
    print("\n🧠 Creating model...")

    if args.restore_path:
        # Load from old checkpoint (fine-tuning)
        print(f"Loading model from: {args.restore_path}")
        model = YourTTSLightningModule.load_from_checkpoint(
            args.restore_path,
            config=config,
            ap=ap,
            tokenizer=tokenizer,
            speaker_manager=speaker_manager,
            language_manager=language_manager,
        )
    else:
        # Create new model
        model = YourTTSLightningModule(
            config=config,
            ap=ap,
            tokenizer=tokenizer,
            speaker_manager=speaker_manager,
            language_manager=language_manager,
        )

    print("✓ Model created")
    print(f"  - Text Encoder layers: {config.model_args.num_layers_text_encoder}")
    print(f"  - Hidden channels: {config.model_args.hidden_channels}")
    print(f"  - Using d-vectors: {config.model_args.use_d_vector_file}")
    print(f"  - Using SCL: {config.model_args.use_speaker_encoder_as_loss}")
    print(f"  - Multilingual: {config.model_args.use_language_embedding}")

    # ========================================
    # Setup Callbacks
    # ========================================
    print("\n⚙️  Setting up callbacks...")

    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_path, "checkpoints"),
        filename="yourtts-{epoch:04d}-{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Early stopping (optional)
    if hasattr(config, 'early_stopping') and config.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 20,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    print(f"✓ Added {len(callbacks)} callbacks")

    # ========================================
    # Setup Logger
    # ========================================
    print("\n📊 Setting up loggers...")

    loggers = []

    # Always use TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=config.output_path,
        name="lightning_logs",
        version=None,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    print(f"✓ TensorBoard logger: {tb_logger.log_dir}")

    # Add WandB logger if requested
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            save_dir=config.output_path,
            log_model=True,  # Log model checkpoints to W&B
        )
        loggers.append(wandb_logger)

        # Log config to W&B
        wandb_logger.experiment.config.update({
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "epochs": config.epochs,
            "learning_rate": config.lr,
            "hidden_channels": config.model_args.hidden_channels,
            "num_layers_text_encoder": config.model_args.num_layers_text_encoder,
            "use_d_vector": config.model_args.use_d_vector_file,
            "use_speaker_encoder_as_loss": config.model_args.use_speaker_encoder_as_loss,
            "multilingual": config.model_args.use_language_embedding,
        })

        print(f"✓ W&B logger initialized")
        print(f"  - Project: {args.wandb_project}")
        print(f"  - Entity: {args.wandb_entity or 'default'}")
        print(f"  - Run name: {args.wandb_run_name or 'auto-generated'}")

    # Use single logger or list of loggers
    logger = loggers if len(loggers) > 1 else loggers[0]

    # ========================================
    # Setup Trainer
    # ========================================
    print("\n🚀 Setting up trainer...")

    # Determine strategy
    strategy = args.strategy
    if strategy == "auto" and args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=args.accelerator,
        devices=args.gpus if args.accelerator == "gpu" else "auto",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        # Note: gradient_clip_val not used - incompatible with manual optimization (GANs)
        # Gradient clipping is handled manually in the model's training_step
        log_every_n_steps=config.print_step if hasattr(config, 'print_step') else 50,
        val_check_interval=1.0,  # Validate once per epoch
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=True,  # Enable cudnn benchmarking for speed
        # Debug mode
        fast_dev_run=5 if args.debug else False,
        # Profiling (optional)
        profiler="simple" if args.debug else None,
    )

    print("✓ Trainer configured")
    print(f"  - Max epochs: {config.epochs}")
    print(f"  - Devices: {args.gpus}")
    print(f"  - Strategy: {strategy}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Gradient clip: {config.grad_clip if hasattr(config, 'grad_clip') else 'None'} (manual)")

    # ========================================
    # Train
    # ========================================
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    try:
        if args.skip_train_epoch:
            # Only run validation
            trainer.validate(model, datamodule=datamodule)
        else:
            # Full training
            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=args.resume_from_checkpoint,
            )

        print("\n" + "="*80)
        print("✓ Training completed successfully!")
        print("="*80)
        print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        print(f"\nTensorBoard logs: {tb_logger.log_dir}")
        print("Run: tensorboard --logdir " + config.output_path)

        if args.use_wandb:
            print(f"\nW&B Run URL: {wandb_logger.experiment.url}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
