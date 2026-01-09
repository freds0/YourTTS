#!/usr/bin/env python3
"""Speech synthesis script for YourTTS Lightning checkpoints.

This script synthesizes speech from text using a trained YourTTS model.

Usage:
    # Basic synthesis
    python examples/synthesize.py \
        --checkpoint outputs_yourtts_ljspeech_test/checkpoints/last.ckpt \
        --text "Hello, this is a test."

    # Save to specific output file
    python examples/synthesize.py \
        --checkpoint outputs_yourtts_ljspeech_test/checkpoints/last.ckpt \
        --text "Hello world!" \
        --output output.wav

    # Batch synthesis from file
    python examples/synthesize.py \
        --checkpoint outputs_yourtts_ljspeech_test/checkpoints/last.ckpt \
        --text_file sentences.txt \
        --output_dir outputs/

    # With speaker/language for multi-speaker models
    python examples/synthesize.py \
        --checkpoint checkpoint.ckpt \
        --text "Hello world!" \
        --speaker_id 0 \
        --language_id 0

    # Using reference audio for voice cloning (if model supports d-vectors)
    python examples/synthesize.py \
        --checkpoint checkpoint.ckpt \
        --text "Hello world!" \
        --reference_audio reference.wav
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits_lightning import YourTTSLightningModule
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize speech using YourTTS Lightning checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt) or old checkpoint (.pth)"
    )

    # Text input (one of these is required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        help="Text to synthesize"
    )
    text_group.add_argument(
        "--text_file",
        type=str,
        help="Path to file with sentences (one per line)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: output.wav)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch synthesis"
    )

    # Model options
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config JSON (optional, uses checkpoint config if not provided)"
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Use GPU for inference"
    )

    # Speaker/Language options
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=None,
        help="Speaker ID for multi-speaker models"
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default=None,
        help="Speaker name for multi-speaker models"
    )
    parser.add_argument(
        "--language_id",
        type=int,
        default=None,
        help="Language ID for multilingual models"
    )
    parser.add_argument(
        "--language_name",
        type=str,
        default=None,
        help="Language name for multilingual models (e.g., 'en', 'pt-br')"
    )

    # Voice cloning options
    parser.add_argument(
        "--reference_audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (requires d-vector model)"
    )

    # Synthesis parameters
    parser.add_argument(
        "--length_scale",
        type=float,
        default=1.0,
        help="Duration scale factor (>1 = slower, <1 = faster)"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.667,
        help="Noise scale for stochastic variation"
    )
    parser.add_argument(
        "--noise_scale_dp",
        type=float,
        default=0.8,
        help="Noise scale for duration predictor"
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    use_cuda: bool = False,
) -> tuple:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config file
        use_cuda: Whether to use GPU

    Returns:
        Tuple of (model, config, ap, tokenizer, speaker_manager, language_manager)
    """
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect checkpoint format
    is_old_format = "model" in checkpoint and "state_dict" not in checkpoint
    is_lightning_format = "state_dict" in checkpoint

    # Get config
    if config_path:
        config = VitsConfig()
        config.load_json(config_path)
        print(f"Loaded config from: {config_path}")
    elif "config" in checkpoint:
        config = checkpoint["config"]
        print("Using config from checkpoint")
    elif "hyper_parameters" in checkpoint and "config" in checkpoint["hyper_parameters"]:
        config = checkpoint["hyper_parameters"]["config"]
        print("Using config from checkpoint hyper_parameters")
    else:
        raise ValueError("No config found. Please provide --config_path")

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize tokenizer
    tokenizer, _ = TTSTokenizer.init_from_config(config)

    # Initialize speaker manager if needed
    speaker_manager = None
    if config.model_args.use_speaker_embedding or config.model_args.use_d_vector_file:
        speaker_manager = SpeakerManager()
        if "speaker_manager" in checkpoint:
            speaker_manager = checkpoint["speaker_manager"]
            print(f"Loaded speaker manager with {speaker_manager.num_speakers} speakers")

    # Initialize language manager if needed
    language_manager = None
    if config.model_args.use_language_embedding:
        language_manager = LanguageManager(config=config)
        if "language_manager" in checkpoint:
            language_manager = checkpoint["language_manager"]
            print(f"Loaded language manager with {language_manager.num_languages} languages")

    # Create model
    model = YourTTSLightningModule(
        config=config,
        ap=ap,
        tokenizer=tokenizer,
        speaker_manager=speaker_manager,
        language_manager=language_manager,
    )

    # Load weights
    if is_old_format:
        print("Loading from old checkpoint format...")
        model.vits_model.load_state_dict(checkpoint["model"], strict=False)
    elif is_lightning_format:
        print("Loading from Lightning checkpoint format...")
        # Load state dict, filtering out non-model keys
        state_dict = checkpoint.get("state_dict", {})
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")

    return model, config, ap, tokenizer, speaker_manager, language_manager


def synthesize_text(
    model: YourTTSLightningModule,
    text: str,
    config: VitsConfig,
    ap: AudioProcessor,
    tokenizer: TTSTokenizer,
    speaker_id: Optional[int] = None,
    speaker_name: Optional[str] = None,
    language_id: Optional[int] = None,
    language_name: Optional[str] = None,
    d_vector: Optional[torch.Tensor] = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_scale_dp: float = 0.8,
    speaker_manager: Optional[SpeakerManager] = None,
    language_manager: Optional[LanguageManager] = None,
) -> np.ndarray:
    """Synthesize speech from text.

    Args:
        model: YourTTS model
        text: Input text
        config: Model config
        ap: Audio processor
        tokenizer: Text tokenizer
        speaker_id: Speaker ID (for multi-speaker models)
        speaker_name: Speaker name (will be converted to ID)
        language_id: Language ID (for multilingual models)
        language_name: Language name (will be converted to ID)
        d_vector: Speaker embedding for voice cloning
        length_scale: Duration scale factor (>1 = slower, <1 = faster)
        noise_scale: Noise scale for decoder variation
        noise_scale_dp: Noise scale for duration predictor
        speaker_manager: Speaker manager for name-to-id conversion
        language_manager: Language manager for name-to-id conversion

    Returns:
        Audio waveform as numpy array
    """
    device = next(model.parameters()).device

    # Convert speaker name to ID if provided
    if speaker_name and speaker_manager:
        speaker_id = speaker_manager.name_to_id.get(speaker_name)
        if speaker_id is None:
            print(f"Warning: Speaker '{speaker_name}' not found. Available: {list(speaker_manager.name_to_id.keys())}")

    # Convert language name to ID if provided
    if language_name and language_manager:
        language_id = language_manager.name_to_id.get(language_name)
        if language_id is None:
            print(f"Warning: Language '{language_name}' not found. Available: {list(language_manager.name_to_id.keys())}")

    # Set inference parameters on the model
    # These are model attributes, not method parameters
    model.vits_model.length_scale = length_scale
    model.vits_model.inference_noise_scale = noise_scale
    model.vits_model.inference_noise_scale_dp = noise_scale_dp

    # Tokenize text
    token_ids = tokenizer.text_to_ids(text)
    tokens = torch.LongTensor(token_ids).unsqueeze(0).to(device)

    # Prepare auxiliary inputs
    aux_input = {
        "x_lengths": torch.LongTensor([len(token_ids)]).to(device),
        "d_vectors": None,
        "speaker_ids": None,
        "language_ids": None,
        "durations": None,
    }

    if speaker_id is not None:
        aux_input["speaker_ids"] = torch.LongTensor([speaker_id]).to(device)
    if language_id is not None:
        aux_input["language_ids"] = torch.LongTensor([language_id]).to(device)
    if d_vector is not None:
        aux_input["d_vectors"] = d_vector.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model.vits_model.inference(
            tokens,
            aux_input=aux_input,
        )

    # Extract waveform
    wav = outputs["model_outputs"]
    wav = wav.squeeze().cpu().numpy()

    return wav


def compute_d_vector(
    reference_audio: str,
    speaker_manager: SpeakerManager,
    ap: AudioProcessor,
) -> torch.Tensor:
    """Compute d-vector from reference audio.

    Args:
        reference_audio: Path to reference audio file
        speaker_manager: Speaker manager with encoder
        ap: Audio processor

    Returns:
        D-vector tensor
    """
    if speaker_manager.encoder is None:
        raise ValueError("Speaker manager does not have an encoder initialized")

    # Load audio
    wav = ap.load_wav(reference_audio)

    # Compute embedding
    d_vector = speaker_manager.compute_embedding_from_clip(wav)

    return torch.FloatTensor(d_vector)


def save_wav(wav: np.ndarray, path: str, sample_rate: int):
    """Save waveform to WAV file using soundfile (librosa's recommended backend).

    Args:
        wav: Audio waveform
        path: Output path
        sample_rate: Sample rate
    """
    import soundfile as sf
    sf.write(path, wav, sample_rate)


def main():
    """Main function."""
    args = parse_args()

    print("\n" + "="*80)
    print("YourTTS Speech Synthesis")
    print("="*80 + "\n")

    # Load model
    model, config, ap, tokenizer, speaker_manager, language_manager = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config_path,
        use_cuda=args.use_cuda,
    )

    # Print model info
    print(f"\nModel configuration:")
    print(f"  Sample rate: {config.audio.sample_rate}")
    print(f"  Multi-speaker: {config.model_args.use_speaker_embedding or config.model_args.use_d_vector_file}")
    print(f"  Multilingual: {config.model_args.use_language_embedding}")
    if speaker_manager:
        print(f"  Speakers: {speaker_manager.num_speakers}")
    if language_manager:
        print(f"  Languages: {language_manager.num_languages}")

    # Compute d-vector if reference audio provided
    d_vector = None
    if args.reference_audio:
        if not config.model_args.use_d_vector_file:
            print("\nWarning: Model does not use d-vectors, reference audio will be ignored")
        elif speaker_manager and speaker_manager.encoder:
            print(f"\nComputing d-vector from: {args.reference_audio}")
            d_vector = compute_d_vector(args.reference_audio, speaker_manager, ap)
            print("D-vector computed successfully")
        else:
            print("\nWarning: Speaker encoder not available, cannot compute d-vector")

    # Get texts to synthesize
    if args.text:
        texts = [args.text]
    else:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(texts)} sentences from {args.text_file}")

    # Determine output paths
    if len(texts) == 1:
        output_path = args.output or "output.wav"
        output_paths = [output_path]
    else:
        output_dir = args.output_dir or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths = [os.path.join(output_dir, f"output_{i:04d}.wav") for i in range(len(texts))]

    # Synthesize
    print("\n" + "-"*80)
    print("Synthesizing...")
    print("-"*80 + "\n")

    for i, (text, output_path) in enumerate(zip(texts, output_paths)):
        print(f"[{i+1}/{len(texts)}] {text[:50]}{'...' if len(text) > 50 else ''}")

        wav = synthesize_text(
            model=model,
            text=text,
            config=config,
            ap=ap,
            tokenizer=tokenizer,
            speaker_id=args.speaker_id,
            speaker_name=args.speaker_name,
            language_id=args.language_id,
            language_name=args.language_name,
            d_vector=d_vector,
            length_scale=args.length_scale,
            noise_scale=args.noise_scale,
            noise_scale_dp=args.noise_scale_dp,
            speaker_manager=speaker_manager,
            language_manager=language_manager,
        )

        # Save audio
        save_wav(wav, output_path, config.audio.sample_rate)
        duration = len(wav) / config.audio.sample_rate
        print(f"    -> Saved: {output_path} ({duration:.2f}s)")

    print("\n" + "="*80)
    print("Synthesis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
