#!/usr/bin/env python3
"""
YourTTS Inference Script
Generates speech from text using a trained checkpoint.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

# Add TTS to path
sys.path.append(str(Path(__file__).parent))

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor


class YourTTSInference:
    """
    Inference class for YourTTS model.
    Handles loading checkpoints and generating speech from text.
    """

    def __init__(self, config_path: str, checkpoint_path: str, use_cuda: bool = True, d_vector_file: str = None):
        """
        Initialize the inference model.

        Args:
            config_path: Path to the configuration file (JSON)
            checkpoint_path: Path to the checkpoint file (.pth or .ckpt)
            use_cuda: Whether to use GPU if available
            d_vector_file: Path to d-vector file (speakers.pth). If None, automatically
                          loads from config.json (model_args.d_vector_file).
        """
        print(f"\n{'='*80}")
        print("YourTTS Inference - Initialization")
        print(f"{'='*80}\n")

        # Setup device
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load config
        print(f"Loading config from: {config_path}")
        self.config = VitsConfig()
        self.config.load_json(config_path)

        # Initialize audio processor
        print("Initializing audio processor...")
        self.ap = AudioProcessor.init_from_config(self.config)

        # Initialize model
        print("Initializing VITS model...")
        self.model = Vits.init_from_config(self.config)
        self.model.to(self.device)

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Set model to eval mode
        self.model.eval()

        # Load d-vectors if available
        self.d_vectors = None
        self.d_vector_file = d_vector_file
        self._load_d_vectors()

        print(f"\n{'='*80}")
        print("Initialization complete!")
        print(f"{'='*80}\n")

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"\nLoading checkpoint from: {checkpoint_path}")

        # PyTorch 2.6+ requires weights_only=False for custom classes
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            # Original TTS checkpoint format
            state_dict = checkpoint['model']
            print("Detected original TTS checkpoint format")
        elif 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint format
            # Extract only model weights (remove 'model.' prefix)
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    state_dict[new_key] = value
            print("Detected PyTorch Lightning checkpoint format")
        else:
            # Direct state dict
            state_dict = checkpoint
            print("Detected direct state dict format")

        # Load state dict
        self.model.load_state_dict(state_dict)
        print("✓ Successfully loaded model weights\n")

    def _load_d_vectors(self):
        """
        Load d-vectors from file.
        Automatically uses d_vector_file from config if not explicitly provided.
        """
        # Determine d-vector file path
        d_vector_path = self.d_vector_file

        if d_vector_path is None:
            # Try to get from config
            if hasattr(self.config, 'model_args') and hasattr(self.config.model_args, 'd_vector_file'):
                d_vector_files = self.config.model_args.d_vector_file
                if d_vector_files and len(d_vector_files) > 0:
                    d_vector_path = d_vector_files[0]
                    print(f"Using d_vector_file from config: {d_vector_path}")

        if d_vector_path is None:
            print("Warning: No d_vector_file specified. Will use random embeddings if needed.")
            return

        if not os.path.exists(d_vector_path):
            print(f"Warning: d_vector_file not found at '{d_vector_path}'. Will use random embeddings if needed.")
            return

        print(f"Loading d-vectors from: {d_vector_path}")

        # Load d-vectors
        try:
            d_vectors_dict = torch.load(d_vector_path, map_location='cpu', weights_only=False)
        except TypeError:
            d_vectors_dict = torch.load(d_vector_path, map_location='cpu')

        # Convert to list of numpy arrays
        self.d_vectors = []
        self.d_vector_names = []

        for name, d_vector_data in d_vectors_dict.items():
            # Handle different d-vector formats
            if isinstance(d_vector_data, dict):
                # If it's a dict, look for 'embedding' or 'vector' key
                if 'embedding' in d_vector_data:
                    d_vector = d_vector_data['embedding']
                elif 'vector' in d_vector_data:
                    d_vector = d_vector_data['vector']
                else:
                    # Skip this entry if we can't find the vector
                    continue
            else:
                d_vector = d_vector_data

            # Convert to numpy if it's a tensor
            if isinstance(d_vector, torch.Tensor):
                d_vector = d_vector.cpu().numpy()

            self.d_vectors.append(d_vector)
            self.d_vector_names.append(name)

        if len(self.d_vectors) > 0:
            print(f"✓ Loaded {len(self.d_vectors)} d-vectors")
            d_vector_shape = self.d_vectors[0].shape if hasattr(self.d_vectors[0], 'shape') else len(self.d_vectors[0])
            print(f"  D-vector dimension: {d_vector_shape}\n")
        else:
            print(f"Warning: No valid d-vectors found in {d_vector_path}\n")
            self.d_vectors = None
            self.d_vector_names = None

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_name: str = None,
        speaker_id: int = None,
        d_vector: np.ndarray = None,
        language_name: str = None,
        language_id: int = None,
        speed: float = 1.0,
        use_random_speaker: bool = True,
    ):
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speaker_name: Speaker name (if using speaker_manager)
            speaker_id: Speaker ID (if not using speaker_manager)
            d_vector: Speaker d-vector embedding (if using d-vectors)
            language_name: Language name (if using language_manager)
            language_id: Language ID (if not using language_manager)
            speed: Speed factor for synthesis (1.0 = normal speed)
            use_random_speaker: If True and no speaker is specified, use a random d-vector

        Returns:
            audio: Generated audio as numpy array
            sample_rate: Sample rate of the audio
        """
        # Process text
        text_inputs = np.asarray(
            self.model.tokenizer.text_to_ids(text, language=language_name),
            dtype=np.int64,
        )

        # Prepare inputs
        text_inputs = torch.from_numpy(text_inputs).unsqueeze(0).to(self.device)

        # Prepare auxiliary inputs
        aux_input = {
            "x_lengths": torch.tensor([text_inputs.shape[1]], dtype=torch.long, device=self.device),
            "d_vectors": None,
            "speaker_ids": None,
            "language_ids": None,
            "durations": None,
        }

        # Handle speaker input
        if speaker_name is not None and hasattr(self.model, 'speaker_manager'):
            # Get speaker ID from name
            speaker_id = self.model.speaker_manager.name_to_id[speaker_name]

        if speaker_id is not None:
            aux_input["speaker_ids"] = torch.tensor([speaker_id], dtype=torch.long, device=self.device)

        # Handle d-vector input
        if d_vector is not None:
            # Use provided d-vector
            # Convert to numpy array if it's a list
            if isinstance(d_vector, list):
                d_vector = np.array(d_vector)
            elif not isinstance(d_vector, np.ndarray):
                d_vector = np.array(d_vector)

            d_vector = torch.from_numpy(d_vector).float().to(self.device)
            if d_vector.dim() == 1:
                d_vector = d_vector.unsqueeze(0)
            aux_input["d_vectors"] = d_vector
        elif use_random_speaker and self.d_vectors is not None and len(self.d_vectors) > 0:
            # Use random d-vector from loaded file
            random_idx = np.random.randint(0, len(self.d_vectors))
            d_vector = self.d_vectors[random_idx]
            print(f"Using random d-vector {random_idx}/{len(self.d_vectors)} ({self.d_vector_names[random_idx]})")

            # Convert to numpy array if it's a list
            if isinstance(d_vector, list):
                d_vector = np.array(d_vector)
            elif not isinstance(d_vector, np.ndarray):
                d_vector = np.array(d_vector)

            d_vector = torch.from_numpy(d_vector).float().to(self.device)
            if d_vector.dim() == 1:
                d_vector = d_vector.unsqueeze(0)
            aux_input["d_vectors"] = d_vector
        elif speaker_id is None and aux_input["speaker_ids"] is None and aux_input["d_vectors"] is None:
            # No speaker info provided, create a default d-vector
            print("Warning: No speaker information provided. Using zero d-vector.")
            d_vector_dim = getattr(self.config.model_args, 'd_vector_dim', 512)
            d_vector = torch.zeros(1, d_vector_dim, dtype=torch.float32, device=self.device)
            aux_input["d_vectors"] = d_vector

        # Handle language input
        if language_name is not None and hasattr(self.model, 'language_manager'):
            language_id = self.model.language_manager.name_to_id[language_name]

        if language_id is not None:
            aux_input["language_ids"] = torch.tensor([language_id], dtype=torch.long, device=self.device)

        # Adjust speed
        original_length_scale = self.model.length_scale
        self.model.length_scale = 1.0 / speed

        # Run inference
        outputs = self.model.inference(text_inputs, aux_input)

        # Restore length scale
        self.model.length_scale = original_length_scale

        # Extract waveform
        waveform = outputs["model_outputs"][0].squeeze().cpu().numpy()

        return waveform, self.config.audio.sample_rate

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speaker_name: str = None,
        speaker_id: int = None,
        d_vector: np.ndarray = None,
        language_name: str = None,
        language_id: int = None,
        speed: float = 1.0,
        use_random_speaker: bool = True,
    ):
        """
        Synthesize speech and save to file.

        Args:
            text: Input text to synthesize
            output_path: Path to save the output audio file
            speaker_name: Speaker name (if using speaker_manager)
            speaker_id: Speaker ID (if not using speaker_manager)
            d_vector: Speaker d-vector embedding (if using d-vectors)
            language_name: Language name (if using language_manager)
            language_id: Language ID (if not using language_manager)
            speed: Speed factor for synthesis (1.0 = normal speed)
            use_random_speaker: If True and no speaker is specified, use a random d-vector
        """
        print(f"\nSynthesizing: '{text}'")

        # Synthesize
        waveform, sample_rate = self.synthesize(
            text=text,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            d_vector=d_vector,
            language_name=language_name,
            language_id=language_id,
            speed=speed,
            use_random_speaker=use_random_speaker,
        )

        # Save audio
        sf.write(output_path, waveform, sample_rate)
        print(f"✓ Audio saved to: {output_path}")
        print(f"  Duration: {len(waveform) / sample_rate:.2f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")

    def list_speakers(self):
        """List all available speakers."""
        if hasattr(self.model, 'speaker_manager') and self.model.speaker_manager is not None:
            print("\nAvailable speakers:")
            for name, idx in self.model.speaker_manager.name_to_id.items():
                print(f"  {idx}: {name}")
        else:
            print("\nNo speaker manager found. Model may not support multi-speaker synthesis.")

    def list_languages(self):
        """List all available languages."""
        if hasattr(self.model, 'language_manager') and self.model.language_manager is not None:
            print("\nAvailable languages:")
            for name, idx in self.model.language_manager.name_to_id.items():
                print(f"  {idx}: {name}")
        else:
            print("\nNo language manager found. Model may not support multi-language synthesis.")

    def list_d_vectors(self):
        """List all loaded d-vectors."""
        if self.d_vectors is not None and len(self.d_vectors) > 0:
            print(f"\nLoaded {len(self.d_vectors)} d-vectors:")
            for idx, name in enumerate(self.d_vector_names):
                print(f"  {idx}: {name}")
        else:
            print("\nNo d-vectors loaded.")

    def get_random_d_vector(self):
        """Get a random d-vector from loaded d-vectors."""
        if self.d_vectors is not None and len(self.d_vectors) > 0:
            random_idx = np.random.randint(0, len(self.d_vectors))
            return self.d_vectors[random_idx], self.d_vector_names[random_idx]
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description='YourTTS Inference - Generate speech from text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (d_vector_file is loaded automatically from config.json)
  python inference.py --config config.json --checkpoint model.pth --text "Hello world" --output output.wav

  # With custom d_vector_file
  python inference.py --config config.json --checkpoint model.pth --text "Hello" --output out.wav --d_vector_file /path/to/speakers.pth

  # With specific speaker
  python inference.py --config config.json --checkpoint model.pth --text "Hello" --output out.wav --speaker_id 0

  # List available speakers
  python inference.py --config config.json --checkpoint model.pth --list_speakers

  # Adjust speed
  python inference.py --config config.json --checkpoint model.pth --text "Hello" --output out.wav --speed 1.2
        """
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (JSON)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pth or .ckpt)'
    )

    # Synthesis arguments
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to synthesize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Output audio file path (default: output.wav)'
    )

    # Speaker/Language arguments
    parser.add_argument(
        '--d_vector_file',
        type=str,
        default=None,
        help='Path to d-vector file (speakers.pth). If not provided, automatically loads from config.json (model_args.d_vector_file).'
    )
    parser.add_argument(
        '--speaker_name',
        type=str,
        default=None,
        help='Speaker name (use --list_speakers to see available speakers)'
    )
    parser.add_argument(
        '--speaker_id',
        type=int,
        default=None,
        help='Speaker ID'
    )
    parser.add_argument(
        '--language_name',
        type=str,
        default=None,
        help='Language name (use --list_languages to see available languages)'
    )
    parser.add_argument(
        '--language_id',
        type=int,
        default=None,
        help='Language ID'
    )

    # Additional arguments
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speed factor (1.0 = normal, >1.0 = faster, <1.0 = slower)'
    )
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        default=True,
        help='Use CUDA if available (default: True)'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA'
    )

    # Info arguments
    parser.add_argument(
        '--list_speakers',
        action='store_true',
        help='List all available speakers and exit'
    )
    parser.add_argument(
        '--list_languages',
        action='store_true',
        help='List all available languages and exit'
    )
    parser.add_argument(
        '--list_d_vectors',
        action='store_true',
        help='List all loaded d-vectors and exit'
    )

    args = parser.parse_args()

    # Handle CUDA flag
    use_cuda = args.use_cuda and not args.no_cuda

    # Initialize inference model
    tts = YourTTSInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_cuda=use_cuda,
        d_vector_file=args.d_vector_file
    )

    # Handle info commands
    if args.list_speakers:
        tts.list_speakers()
        return

    if args.list_languages:
        tts.list_languages()
        return

    if args.list_d_vectors:
        tts.list_d_vectors()
        return

    # Check if text is provided
    if args.text is None:
        print("\nError: --text argument is required for synthesis")
        print("Use --list_speakers or --list_languages to see available options")
        print("Use --help for more information")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Synthesize
    tts.synthesize_to_file(
        text=args.text,
        output_path=args.output,
        speaker_name=args.speaker_name,
        speaker_id=args.speaker_id,
        language_name=args.language_name,
        language_id=args.language_id,
        speed=args.speed,
    )

    print(f"\n{'='*80}")
    print("Inference complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
