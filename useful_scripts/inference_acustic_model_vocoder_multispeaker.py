import argparse
import os
import sys
TTS_PATH = "TTS/"
# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally
import time
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
from TTS.tts.utils.synthesis import synthesis
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping

from TTS.vocoder.models import setup_model as vocoder_setup_model

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_vocoder(vocoder_checkpoint_filepath, vocoder_config_path):
    # Load vocoder moel config
    vocoder_config = load_config(vocoder_config_path)
    # Load vocoder checkpoint
    vocoder_model = vocoder_setup_model(vocoder_config)
    vocoder_model.load_state_dict(torch.load(vocoder_checkpoint_filepath, map_location="cpu")["model"])
    if USE_CUDA:
        vocoder_model = vocoder_model.cuda()

    return vocoder_model


def load_model_config(checkpoint_filepath, config_filepath, speakers_filepath) -> tuple:
    # load acustic model config
    config = load_config(config_filepath)
    config['speakers_file'] = speakers_filepath
    model = setup_model(config)
    # Load acustic model checkpoint
    cp = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
    model_weights = cp['model'].copy()
    for key in list(model_weights.keys()):
        if "speaker_encoder" in key:
            del model_weights[key]
        if "audio_transform" in key:
            del model_weights[key]

    model.load_state_dict(model_weights)
    model.eval()

    if USE_CUDA:
        model = model.cuda()

    return model, config


def save_file(filename, audio, output_type='wav', sampling_rate=22050) -> None:
    '''
    Save wav or ogg output file
    Args:
        filename: filename without extension (.wav or .ogg)
        audio: numpy data referent to waveform
        output_type: wav or ogg
        sampling_rate: examples 22050, 44100

    Returns: None
    '''
    if output_type == 'wav':
        sf.write(filename + '.wav', audio, sampling_rate, 'PCM_16')
    else:
        # A bug occurred in the soundfile lib: cuts the end of the ogg file.
        sf.write(filename + '.ogg', audio, sampling_rate, format='ogg', subtype='vorbis')


def add_pause(full_audio, pause, default_pause, sample_rate):
    '''
    Adds a pause at the end of an audio file.
    '''
    pause = int(default_pause * sample_rate) if pause == -1 else int(pause * sample_rate / 1000)
    full_audio = np.append(full_audio, np.zeros(pause, dtype=np.int16))
    return full_audio


def synthesize_waveform(model, config, vocoder_model, sentence, speaker_name=None, use_griffin_lim=False, use_cuda=True, do_trim_silence=False)-> np.array:
    '''
    Run inference on the model.
    '''
    if (speaker_name):
        speaker_id = model.speaker_manager.ids[speaker_name + "\n"]
    else:
        speaker_id = None

    waveform, alignments, text_input, outputs = synthesis(
                                                    model,
                                                    text = sentence,
                                                    CONFIG = config,
                                                    use_cuda = use_cuda,
                                                    speaker_id = speaker_id,
                                                    style_wav = None,
                                                    style_text = None,
                                                    use_griffin_lim = use_griffin_lim,
                                                    do_trim_silence = False,
                                                    d_vector = None,
                                                    language_id = None
                                                  ).values()

    if (not use_griffin_lim):
        # Using vocoder to waveform generation
        mel_postnet_spec = outputs["model_outputs"].transpose(1, 2)
        vocoder_input = mel_postnet_spec

        vocoder_input = vocoder_input.to('cpu')
        vocoder_model = vocoder_model.to('cpu')
        waveform = vocoder_model.inference(vocoder_input).squeeze(0).squeeze(0).cpu().numpy()

    return waveform


def generate_wavfile(model, config, vocoder_model, sentences, speaker_name, output_folder, sr=22050, audio_format='wav') -> None:
    '''
    Run inference on the model and save a wav file
    '''
    #full_audio = np.empty((0,1), dtype=np.int16)
    #pause = 0.1
    for index, sentence in enumerate(tqdm(sentences)):
        waveform = synthesize_waveform(model, config, vocoder_model, sentence, speaker_name)
        '''
        if i + 1 != len(sentences):
            full_audio = add_pause(waveform, pause, pause_sec, sr)
        '''
        filename = "output-{0:04d}".format(index)
        filepath = os.path.join(output_folder, filename)
        save_file(filepath, waveform, audio_format, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    # Acustic model filepaths
    parser.add_argument('--checkpoint', default='/home/fred/tacotron2_multispeaker_with_brspeech_best-October-07-2022_01+04AM-0cbaa0f6/checkpoint_2180000.pth', help='Checkpoint pth filepath')
    parser.add_argument('--speaker',
                        default='/home/fred/tacotron2_multispeaker_with_brspeech_best-October-07-2022_01+04AM-0cbaa0f6/speakers.pth',
                        help='Speaker pth filepath')
    parser.add_argument('--config', default='/home/fred/tacotron2_multispeaker_with_brspeech_best-October-07-2022_01+04AM-0cbaa0f6/config.json', help='Config json filepath')
    # Vocoder model filepaths
    parser.add_argument('--checkpoint_vocoder', default='/home/fred/hifi_September-28-2022_10+51AM-0cbaa0f6/checkpoint_550000.pth', help='Checkpoint pth filepath')
    parser.add_argument('--config_vocoder', default='/home/fred/hifi_September-28-2022_10+51AM-0cbaa0f6/config.json', help='Config json filepath')
    parser.add_argument('--speaker_name', default='', help='Speaker Name')
    parser.add_argument('--input_file', default='sentences.txt', help='Sentences input file')
    parser.add_argument('--output_folder', default='output_tacotron2hifigan', help='Output folder')
    parser.add_argument('--sr', default=22050)
    parser.add_argument('--audio_format', default='wav')
    args = parser.parse_args()

    with open(args.input_file) as f:
        sentences = f.readlines()

    output_folder = os.path.join(args.base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    model, config = load_model_config(args.checkpoint, args.config, args.speaker)
    vocoder_model = load_vocoder(args.checkpoint_vocoder, args.config_vocoder)
    generate_wavfile(model, config, vocoder_model, sentences, args.speaker_name, args.output_folder, sr=22050, audio_format='wav')


if __name__ == "__main__":
    main()
