# YourTTS

This repository contains a refactored version of [Coqui TTS](https://github.com/coqui-ai/TTS) focused exclusively on the **YourTTS** model for multilingual, multi-speaker Text-to-Speech with zero-shot voice cloning.

YourTTS is a VITS-based model that uses speaker embeddings (d-vectors) for voice cloning and supports multiple languages.

## ⚠️ Important Notes

- **This repository only includes the YourTTS model** - other TTS models from Coqui TTS have been removed
- **Python 3.9 is required** - This version is specifically configured for Python 3.9
- **Bangla (Bengali) language is not supported** due to library compatibility issues with Python 3.9

## Paper

[YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone](https://arxiv.org/abs/2112.02418)

```bibtex
@inproceedings{casanova2022yourtts,
  title={YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone},
  author={Casanova, Edresson and Weber, Julian and Shulby, Christopher and Junior, Arnaldo Candido and Gölge, Eren and Ponti, Moacir A},
  booktitle={International Conference on Machine Learning},
  pages={2709--2720},
  year={2022},
  organization={PMLR}
}
```

## Requirements

- **Python 3.9** (required)
- CUDA-compatible GPU (optional, for faster inference and training)

## Installation

### Using Conda (Recommended)

```bash
# Create Python 3.9 environment
conda create -n yourtts python=3.9
conda activate yourtts

# Clone the repository
git clone https://github.com/Edresson/YourTTS.git
cd YourTTS

# Install dependencies
pip install -e .
```

### Using venv

```bash
# Ensure you have Python 3.9 installed
python3.9 -m venv yourtts_env
source yourtts_env/bin/activate  # On Windows: yourtts_env\Scripts\activate

# Clone the repository
git clone https://github.com/Edresson/YourTTS.git
cd YourTTS

# Install dependencies
pip install -e .
```

## Quick Start - Inference

### Python API

```python
from TTS.api import TTS

# Initialize YourTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)

# Generate speech with voice cloning
tts.tts_to_file(
    text="Hello, this is a test.",
    speaker_wav="path/to/reference/audio.wav",
    language="en",
    file_path="output.wav"
)
```

### Command Line Interface

```bash
# Synthesize with voice cloning
tts --text "Hello, this is a test." \
    --model_name "tts_models/multilingual/multi-dataset/your_tts" \
    --speaker_wav "path/to/reference/audio.wav" \
    --language_idx "en" \
    --out_path "output.wav"
```

## Training

This repository includes training recipes for YourTTS:

### Single-language (VCTK)

```bash
cd recipes/vctk/yourtts/
python train_yourtts.py
```

### Multilingual

```bash
cd recipes/multilingual/cml_yourtts/
python train_yourtts.py
```

## Computing Speaker Embeddings

YourTTS requires pre-computed speaker embeddings (d-vectors) for voice cloning:

```bash
python TTS/bin/compute_embeddings.py \
    --model_path path/to/speaker_encoder/model.pth \
    --config_path path/to/speaker_encoder/config.json \
    --output_path embeddings.pth \
    --dataset_path path/to/dataset
```

## Model Architecture

**YourTTS** is based on the VITS (Variational Inference with adversarial learning for end-to-end TTS) architecture with the following key components:

- **TTS Model**: VITS with speaker conditioning via d-vectors
- **Speaker Encoder**: Neural encoder for computing speaker embeddings
- **Vocoder**: HiFiGAN for mel-spectrogram to waveform conversion

## Supported Languages

The pre-trained YourTTS model supports multiple languages including:
- English (en)
- Portuguese (pt)
- French (fr)
- Spanish (es)
- German (de)
- Italian (it)
- Chinese (zh)
- Korean (ko)
- Japanese (ja)
- Belarusian (be)
- And more...

**Note:** Bangla (Bengali) is not supported in this Python 3.9 version due to library compatibility issues.

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0).

YourTTS model is licensed under CC BY-NC-ND 4.0.

## Citation

If you use this code or the YourTTS model, please cite:

```bibtex
@inproceedings{casanova2022yourtts,
  title={YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone},
  author={Casanova, Edresson and Weber, Julian and Shulby, Christopher and Junior, Arnaldo Candido and Gölge, Eren and Ponti, Moacir A},
  booktitle={International Conference on Machine Learning},
  pages={2709--2720},
  year={2022},
  organization={PMLR}
}
```

## Differences from Original Coqui TTS

This repository is a streamlined version of [Coqui TTS](https://github.com/coqui-ai/TTS) with the following key differences:

- **Only YourTTS model included** - All other TTS models have been removed to reduce complexity
- **Python 3.9 only** - Optimized specifically for Python 3.9 compatibility
- **Reduced dependencies** - Only essential dependencies for YourTTS are included
- **No Bangla support** - Bengali language phonemizer removed due to Python 3.9 incompatibility

For the full Coqui TTS with all models and broader Python version support, please visit the [original repository](https://github.com/coqui-ai/TTS).

## Contact

For questions and issues, please open an issue on GitHub.
