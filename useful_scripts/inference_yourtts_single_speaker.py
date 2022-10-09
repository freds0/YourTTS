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

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_config(checkpoint_filepath, config_filepath) -> tuple:
    # load the config
    config = load_config(config_filepath)
    # load the audio processor
    #ap = AudioProcessor(**config.audio)
    #config.model_args['d_vector_file'] = speakers_embeddings_filepath
    #config.model_args['use_speaker_encoder_as_loss'] = False
    model = setup_model(config)

    cp = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
    # remove speaker encoder
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

    #model.noise_scale = 0.0  # defines the noise variance applied to the random z vector at inference.
    #model.length_scale = 1.0  # scaler for the duration predictor. The larger it is, the slower the speech.
    #model.noise_scale_w = 1.0 # defines the noise variance applied to the duration predictor z vector at inference.

    model.length_scale = 1.7  # scaler for the duration predictor. The larger it is, the slower the speech.
    model.inference_noise_scale = 0.333 # defines the noise variance applied to the random z vector at inference.
    model.inference_noise_scale_dp = 0.333 # defines the noise variance applied to the duration predictor z vector at inference.

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


def synthesize_waveform(model, config, sentence, use_griffin_lim = True, do_trim_silence = False)-> np.array:
    '''
    Run inference on the model.
    '''
    waveform, alignment, _, _ = synthesis(
                              model,
                              sentence,
                              config,
                              DEVICE,
                              speaker_id = None,
                              use_griffin_lim = use_griffin_lim,
                              do_trim_silence = do_trim_silence
    ).values()
    return waveform

def generate_wavfile(model, config, sentences, output_folder, sr=22050, audio_format='wav') -> None:
    '''
    Run inference on the model and save a wav file
    '''
    #full_audio = np.empty((0,1), dtype=np.int16)
    #pause = 0.1
    for index, sentence in enumerate(tqdm(sentences)):
        waveform = synthesize_waveform(model, config, sentence)
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
    parser.add_argument('--checkpoint', default='/home/fred/Projetos/YourTTS/checkpoints/teste_your_tts_wagner_portal/vits_tts-portuguese_titanet_wagner_portal-July-26-2022_06+32PM-0cbaa0f6/checkpoint_400650.pth', help='Checkpoint pth filepath')
    parser.add_argument('--config', default='/home/fred/Projetos/YourTTS/checkpoints/teste_your_tts_wagner_portal/vits_tts-portuguese_titanet_wagner_portal-July-26-2022_06+32PM-0cbaa0f6/config.json', help='Config json filepath')
    parser.add_argument('--input_file', default='sentences.txt', help='Sentences input file')
    parser.add_argument('--output_folder', default='output', help='Output folder')
    parser.add_argument('--sr', default=22050)
    parser.add_argument('--audio_format', default='wav')
    args = parser.parse_args()

    with open(args.input_file) as f:
        sentences = f.readlines()

    output_folder = os.path.join(args.base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    model, config = load_model_config(args.checkpoint, args.config)
    generate_wavfile(model, config, sentences, args.output_folder, sr=22050, audio_format='wav')

if __name__ == "_
_main__":
    main()
