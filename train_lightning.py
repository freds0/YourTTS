#!/usr/bin/env python3
"""
YourTTS Training Script using PyTorch Lightning
Alternative training implementation that replicates the original training logic.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add TTS to path
sys.path.append(str(Path(__file__).parent))

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.layers.losses import VitsGeneratorLoss, VitsDiscriminatorLoss
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor


class VitsLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VITS model.
    Maintains the exact same training logic as the original implementation.
    """

    def __init__(self, config: VitsConfig, ap: AudioProcessor):
        super().__init__()
        self.config = config
        self.ap = ap

        # Initialize VITS model (reuses existing implementation)
        self.model = Vits.init_from_config(config)

        # Initialize loss functions (criterion)
        self.criterion_disc = VitsDiscriminatorLoss(config)
        self.criterion_gen = VitsGeneratorLoss(config)
        self.criterion = [self.criterion_disc, self.criterion_gen]

        # Manual optimization (dual optimizers)
        self.automatic_optimization = False

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'ap'])

    def configure_optimizers(self):
        """
        Configure two optimizers: one for discriminator (idx=0), one for generator (idx=1).
        This replicates the original two-optimizer training strategy.
        """
        # Optimizer parameters from config
        optimizer_name = getattr(self.config, 'optimizer', 'AdamW')
        optimizer_params = getattr(self.config, 'optimizer_params', {
            'betas': [0.8, 0.99],
            'eps': 1e-9,
            'weight_decay': 0.01
        })

        # Discriminator parameters
        disc_params = []
        # Generator parameters (everything except discriminator)
        gen_params = []

        for name, param in self.model.named_parameters():
            if 'disc' in name:
                disc_params.append(param)
            else:
                gen_params.append(param)

        # Create optimizers
        optimizer_class = getattr(torch.optim, optimizer_name)

        # optimizer_idx=0: Discriminator
        optimizer_disc = optimizer_class(
            disc_params,
            lr=self.config.lr_disc,
            **optimizer_params
        )

        # optimizer_idx=1: Generator
        optimizer_gen = optimizer_class(
            gen_params,
            lr=self.config.lr_gen,
            **optimizer_params
        )

        # Learning rate schedulers
        scheduler_gen_name = getattr(self.config, 'lr_scheduler_gen', 'ExponentialLR')
        scheduler_gen_params = getattr(self.config, 'lr_scheduler_gen_params', {
            'gamma': 0.999875,
            'last_epoch': -1
        })

        scheduler_disc_name = getattr(self.config, 'lr_scheduler_disc', 'ExponentialLR')
        scheduler_disc_params = getattr(self.config, 'lr_scheduler_disc_params', {
            'gamma': 0.999875,
            'last_epoch': -1
        })

        scheduler_gen_class = getattr(torch.optim.lr_scheduler, scheduler_gen_name)
        scheduler_disc_class = getattr(torch.optim.lr_scheduler, scheduler_disc_name)

        scheduler_gen = scheduler_gen_class(optimizer_gen, **scheduler_gen_params)
        scheduler_disc = scheduler_disc_class(optimizer_disc, **scheduler_disc_params)

        # Return [disc, gen] to match optimizer_idx order
        return [optimizer_disc, optimizer_gen], [scheduler_disc, scheduler_gen]

    def training_step(self, batch, batch_idx):
        """
        Training step with dual optimizer strategy.
        Replicates the original two-pass training approach using train_step from VITS model.
        """
        optimizer_disc, optimizer_gen = self.optimizers()

        # Format batch (compute speaker IDs, d-vectors, language IDs)
        batch = self.model.format_batch(batch)

        # Format batch on device (compute spectrograms)
        batch = self.model.format_batch_on_device(batch)

        # ========== DISCRIMINATOR PASS (optimizer_idx=0) ==========
        # This computes discriminator loss and caches outputs
        outputs_disc, loss_dict_disc = self.model.train_step(
            batch, self.criterion, optimizer_idx=0
        )

        # Backward and optimize discriminator
        optimizer_disc.zero_grad()
        loss_disc = loss_dict_disc['loss']
        self.manual_backward(loss_disc)

        # Gradient clipping for discriminator
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip[1] > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if 'disc' in str(p)],
                self.config.grad_clip[1]
            )

        optimizer_disc.step()

        # ========== GENERATOR PASS (optimizer_idx=1) ==========
        # This uses cached outputs from discriminator pass
        outputs_gen, loss_dict_gen = self.model.train_step(
            batch, self.criterion, optimizer_idx=1
        )

        # Backward and optimize generator
        optimizer_gen.zero_grad()
        loss_gen = loss_dict_gen['loss']
        self.manual_backward(loss_gen)

        # Gradient clipping for generator
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip[0] > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if 'disc' not in str(p)],
                self.config.grad_clip[0]
            )

        optimizer_gen.step()

        # Learning rate schedulers
        scheduler_disc, scheduler_gen = self.lr_schedulers()
        scheduler_disc.step()
        scheduler_gen.step()

        # Total loss for logging
        total_loss = loss_disc + loss_gen

        # Logging
        self.log('train/loss_total', total_loss, prog_bar=True)
        self.log('train/loss_disc', loss_disc, prog_bar=True)
        self.log('train/loss_gen', loss_gen, prog_bar=False)

        # Log individual loss components
        for key, value in loss_dict_disc.items():
            if key != 'loss':
                self.log(f'train/disc_{key}', value)

        for key, value in loss_dict_gen.items():
            if key != 'loss':
                self.log(f'train/gen_{key}', value)

        # Log learning rates
        self.log('train/lr_disc', optimizer_disc.param_groups[0]['lr'])
        self.log('train/lr_gen', optimizer_gen.param_groups[0]['lr'])

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step using the model's eval_step method.
        """
        # Format batch (compute speaker IDs, d-vectors, language IDs)
        batch = self.model.format_batch(batch)

        # Format batch on device (compute spectrograms)
        batch = self.model.format_batch_on_device(batch)

        # Run discriminator pass
        outputs_disc, loss_dict_disc = self.model.eval_step(
            batch, self.criterion, optimizer_idx=0
        )

        # Run generator pass
        outputs_gen, loss_dict_gen = self.model.eval_step(
            batch, self.criterion, optimizer_idx=1
        )

        # Total loss
        loss_disc = loss_dict_disc['loss']
        loss_gen = loss_dict_gen['loss']
        total_loss = loss_disc + loss_gen

        # Logging
        self.log('val/loss_total', total_loss, prog_bar=True)
        self.log('val/loss_disc', loss_disc)
        self.log('val/loss_gen', loss_gen)

        # Log individual loss components
        for key, value in loss_dict_disc.items():
            if key != 'loss':
                self.log(f'val/disc_{key}', value)

        for key, value in loss_dict_gen.items():
            if key != 'loss':
                self.log(f'val/gen_{key}', value)

        return total_loss


def create_dataloaders(config: VitsConfig, ap: AudioProcessor, model: Vits):
    """
    Create train and validation dataloaders.
    Uses the existing dataset implementation.

    Args:
        config: VitsConfig object
        ap: AudioProcessor object
        model: Vits model (needed for tokenizer, speaker_manager, etc.)
    """
    # Load samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size
    )

    # Create datasets using the model's get_data_loader method
    # But since we're in Lightning, we'll create them manually
    from TTS.tts.models.vits import VitsDataset

    train_dataset = VitsDataset(
        model_args=config.model_args,
        samples=train_samples,
        batch_group_size=config.batch_group_size * config.batch_size,
        min_text_len=config.min_text_len,
        max_text_len=config.max_text_len,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        phoneme_cache_path=config.phoneme_cache_path,
        precompute_num_workers=config.precompute_num_workers,
        verbose=True,
        tokenizer=model.tokenizer,  # Use model's tokenizer
        start_by_longest=config.start_by_longest,
    )

    eval_dataset = VitsDataset(
        model_args=config.model_args,
        samples=eval_samples,
        batch_group_size=0,  # No grouping for eval
        min_text_len=config.min_text_len,
        max_text_len=config.max_text_len,
        min_audio_len=config.min_audio_len,
        max_audio_len=config.max_audio_len,
        phoneme_cache_path=config.phoneme_cache_path,
        precompute_num_workers=config.precompute_num_workers,
        verbose=False,
        tokenizer=model.tokenizer,  # Share tokenizer
        start_by_longest=False,
    )

    # Preprocess samples (sort, filter, etc.)
    train_dataset.preprocess_samples()
    eval_dataset.preprocess_samples()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_loader_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_loader_workers,
        collate_fn=eval_dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, eval_loader


def train(config_path: str, output_dir: str = './outputs_lightning', restore_path: str = None):
    """
    Main training function using PyTorch Lightning.

    Args:
        config_path: Path to the configuration file (JSON)
        output_dir: Directory to save outputs
        restore_path: Path to checkpoint to restore from (optional)
    """
    # Load config
    config = VitsConfig()
    config.load_json(config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize base VITS model first (needed for tokenizer)
    # We'll wrap it in Lightning module later
    base_model = Vits.init_from_config(config)

    # Create dataloaders (needs model for tokenizer)
    train_loader, eval_loader = create_dataloaders(config, ap, base_model)

    # Initialize Lightning module (wraps the base model)
    model = VitsLightningModule(config, ap)
    # Replace the model with the already-initialized base_model (which has tokenizer, etc.)
    model.model = base_model

    if restore_path:
        print(f"\n{'='*80}")
        print(f"Loading checkpoint from: {restore_path}")
        print(f"{'='*80}\n")

        # Load state dict from checkpoint
        checkpoint = torch.load(restore_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            # Original TTS checkpoint format
            model.model.load_state_dict(checkpoint['model'])
            print("✓ Loaded model weights from 'model' key")
        elif 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint format
            model.load_state_dict(checkpoint['state_dict'])
            print("✓ Loaded model weights from 'state_dict' key")
        else:
            # Direct state dict
            model.model.load_state_dict(checkpoint)
            print("✓ Loaded model weights directly")

        print(f"✓ Successfully restored model from checkpoint")
        print(f"{'='*80}\n")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='vits-{epoch:03d}-{train/loss_total:.4f}',
        save_top_k=config.save_n_checkpoints,
        monitor='train/loss_total',
        mode='min',
        every_n_train_steps=config.save_step,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='lightning_logs',
        version=config.run_name if hasattr(config, 'run_name') else None,
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config.print_step,
        val_check_interval=config.eval_step if hasattr(config, 'eval_step') else 1.0,
        gradient_clip_val=None,  # Manual gradient clipping in training_step
        precision='16-mixed' if config.mixed_precision else '32',
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    # Note: If using a Lightning checkpoint (.ckpt), you can also resume with ckpt_path parameter
    # But for original TTS checkpoints (.pth), we load them manually above
    trainer.fit(model, train_loader, eval_loader)

    return trainer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YourTTS with PyTorch Lightning')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (JSON)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs_lightning',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--restore_path',
        type=str,
        default=None,
        help='Path to checkpoint file to restore from (.pth or .ckpt)'
    )

    args = parser.parse_args()

    train(args.config, args.output_dir, args.restore_path)
