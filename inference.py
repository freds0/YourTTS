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

speaker_idx = {
    '10107': 0,
    '10199': 1,
    '10670': 2,
    '11247': 3,
    '11995': 4,
    '12249': 5,
    '12287': 6,
    '12428': 7,
    '12626': 8,
    '12670': 9,
    '12707': 10,
    '12710': 11,
    '12865': 12,
    '13063': 13,
    '13069': 14,
    '13196': 15,
    '2959': 16,
    '2961': 17,
    '3037': 18,
    '3050': 19,
    '3369': 20,
    '3427': 21,
    '3718': 22,
    '3814': 23,
    '3976': 24,
    '4000': 25,
    '4067': 26,
    '4341': 27,
    '4367': 28,
    '4405': 29,
    '4778': 30,
    '4783': 31,
    '4801': 32,
    '5025': 33,
    '5068': 34,
    '5103': 35,
    '533': 36,
    '5417': 37,
    '5677': 38,
    '5705': 39,
    '5739': 40,
    '5888': 41,
    '6187': 42,
    '6207': 43,
    '6549': 44,
    '6566': 45,
    '6581': 46,
    '6601': 47,
    '6700': 48,
    '7028': 49,
    '7407': 50,
    '7925': 51,
    '8680': 52,
    '9056': 53,
    '9217': 54,
    '9351': 55,
    '9485': 56
}
def load_model_config(checkpoint_filepath, config_filepath, speakers_embeddings_filepath) -> tuple:
    # load the config
    config = load_config(config_filepath)
    # load the audio processor
    #ap = AudioProcessor(**config.audio)
    config.model_args['d_vector_file'] = speakers_embeddings_filepath
    config.model_args['use_speaker_encoder_as_loss'] = False
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

    model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
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

def extract_reference_embedding(speaker_embedding_filepath, speaker_id):
    '''
    Given a dictionary containing the embeddings of the speakers, this function finds an embedding of a given speaker_id using regular expression.
    '''
    import re
    speaker_mapping_dict = load_speaker_mapping(speaker_embedding_filepath)

    def find_matches(d, item):
        for k in d:
            if re.match('^'+item, k):
                return d[k]['embedding']

    speaker_re = f"{speaker_id}*"
    return find_matches(speaker_mapping_dict, speaker_re)

def synthesize_waveform(model, config, sentence, speaker_id, reference_emb, use_griffin_lim = True, do_trim_silence = False)-> np.array:
    '''
    Run inference on the model.
    '''
    waveform, alignment, _, _ = synthesis(
                              model,
                              sentence,
                              config,
                              DEVICE,
                              speaker_id = speaker_id,
                              d_vector = reference_emb,
                              use_griffin_lim = use_griffin_lim,
                              do_trim_silence = do_trim_silence
    ).values()
    return waveform

def generate_wavfile(model, config, sentences, speaker_id, speakers_embeddings_filepath, output_folder, sr=22050, audio_format='wav') -> None:
    '''
    Run inference on the model and save a wav file
    '''
    reference_emb = extract_reference_embedding(speakers_embeddings_filepath, speaker_id)
    speaker_id = speaker_idx[speaker_id]
    #full_audio = np.empty((0,1), dtype=np.int16)
    #pause = 0.1
    for index, sentence in enumerate(tqdm(sentences)):
        waveform = synthesize_waveform(model, config, sentence, speaker_id, reference_emb)
        '''
        if i + 1 != len(sentences):
            full_audio = add_pause(waveform, pause, pause_sec, sr)
        '''
        filename = f"output-{index}"
        filepath = os.path.join(output_folder, filename)
        save_file(filepath, waveform, audio_format, sr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--checkpoint', default='./checkpoints/checkpoints-coqui-tts-dev/checkpoint_125000.pth', help='Checkpoint pth filepath')
    parser.add_argument('--config', default='./checkpoints/checkpoints-coqui-tts-dev/config.json', help='Config json filepath')
    parser.add_argument('--speaker_embeddings', default='./speaker_embeddings/d_vector_file.json', help='Speaker embeddings json filepath')
    parser.add_argument('--speaker', default='6207', help="Speaker's dataset id")
    parser.add_argument('--input_file', default='sentences.txt', help='Sentences input file')
    parser.add_argument('--output_folder', default='output', help='Output folder')
    parser.add_argument('--sr', default=22050)
    parser.add_argument('--audio_format', default='wav')
    args = parser.parse_args()

    with open(args.input_file) as f:
        sentences = f.readlines()

    output_folder = os.path.join(args.base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    model, config = load_model_config(args.checkpoint, args.config, args.speaker_embeddings)
    generate_wavfile(model, config, sentences, args.speaker, args.speaker_embeddings, args.output_folder, sr=22050, audio_format='wav')

if __name__ == "__main__":
    main()
