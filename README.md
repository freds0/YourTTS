# YourTTS <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="16"/>

This is a implementation of [YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone](https://arxiv.org/abs/2112.02418). 

This code is a fork of [https://github.com/Edresson/Coqui-TTS](https://github.com/Edresson/Coqui-TTS).   

A complete documentation can be found at [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS).

## Install
🐸TTS is tested on Ubuntu 18.04 with **python >= 3.6, < 3.9**.

If you are only interested in train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/freds0/YourTTS/
# For cuda 10.2
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
# or for cuda 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

cd YourTTS
pip install -e .
```

If you are on Ubuntu (Debian), you can also run following commands for installation.

```bash
$ make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a diffent OS.
$ make install
```

## Download Checkpoints and Configs

We can download the latest checkpoint and configs from [Coqui released model](https://github.com/coqui-ai/TTS/releases/download/v0.5.0_models/tts_models--multilingual--multi-dataset--your_tts.zip).

Alternatively, you can synthesize some text: 

```bash
tts  --text "This is an example!" --model_name tts_models/multilingual/multi-dataset/your_tts  --speaker_wav target_speaker_wav.wav --language_idx "en"
```

where "target_speaker_wav.wav" is a reference file. This way the checkpoints and configs will be downloaded to the directory:

```bash
ls ~/.local/shared/tts/ 
```

## Training

Before carrying out the training, it will first be necessary to extract the embeddings from the speakers. To do this use the following command:

```bash
python3 TTS/bin/compute_embeddings.py model_se.pth.tar config_se.json config.json ./d_vector_file.json
```
The files "model_se.pth.tar", "config_se.json" and "config.json" can be found it [Coqui released model](https://github.com/coqui-ai/TTS/releases/download/v0.5.0_models/tts_models--multilingual--multi-dataset--your_tts.zip). In the "config.json" file you will need set the paths. The results will be saved at "./d_vector_file.json".

For training YourTTS you must 
In "d_vector_file" you need to set in "config.json" the path to the speaker embeddings file ("./d_vector_file.json"). 

To perform the training, configure the paths correctly in the "config.json" file and run the command:


```bash 
python3 TTS/bin/train_tts.py --config_path config.json
```

## Synthesis

### TTS
To use the frog TTS released YourTTS model for Text-to-Speech use the following command:

```bash
tts  --text "This is an example!" --model_name tts_models/multilingual/multi-dataset/your_tts  --speaker_wav target_speaker_wav.wav --language_idx "en"
```


### Voice Conversion

To use the TTS released YourTTS model for voice conversion use the following command:

```bash
tts --model_name tts_models/multilingual/multi-dataset/your_tts  --speaker_wav source_speaker_wav.wav --reference_wav  target_content_wav.wav --language_idx "en"
```
Considering the "target_content_wav.wav" as the reference wave file to convert into the voice of the "source_speaker_wav.wav" speaker.
