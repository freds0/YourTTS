#!/usr/bin/env python3
"""
Script to compute speaker embeddings for LJSpeech dataset
"""

from TTS.bin.compute_embeddings import compute_embeddings

# Configuration
SPEAKER_ENCODER_CHECKPOINT = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
DATASET_PATH = "/home/fred/Projetos/DATASETS/LJSpeech-1.1/"
OUTPUT_PATH = f"{DATASET_PATH}/speakers.pth"

print("Computing speaker embeddings for LJSpeech...")
print(f"Dataset path: {DATASET_PATH}")
print(f"Output: {OUTPUT_PATH}")
print()

compute_embeddings(
    SPEAKER_ENCODER_CHECKPOINT,
    SPEAKER_ENCODER_CONFIG,
    OUTPUT_PATH,
    old_speakers_file=None,
    config_dataset_path=None,
    formatter_name="ljspeech",
    dataset_name="ljspeech",
    dataset_path=DATASET_PATH,
    meta_file_train="metadata.csv",
    meta_file_val=None,
    disable_cuda=False,
    no_eval=True,
)

print()
print("✓ Embeddings computed successfully!")
print(f"✓ Saved to: {OUTPUT_PATH}")

# Verify format
import torch
embeddings = torch.load(OUTPUT_PATH)
print(f"✓ Total embeddings: {len(embeddings)}")
print(f"✓ Sample keys: {list(embeddings.keys())[:3]}")
