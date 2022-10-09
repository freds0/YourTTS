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
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager

try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
#from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping

#USE_CUDA = torch.cuda.is_available()
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_config(checkpoint_filepath, config_filepath, speakers_embeddings_filepath, language_embeddings_filepath, use_cuda = True) -> tuple:
    # load the config
    config = load_config(config_filepath)

    config.model_args['d_vector_file'] = speakers_embeddings_filepath
    config.model_args['use_speaker_encoder_as_loss'] = False

    model = setup_model(config)
    model.language_manager = LanguageManager(language_embeddings_filepath)
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

    if use_cuda:
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

'''
def extract_reference_embedding(speaker_embedding_filepath, speaker_id):

    Given a dictionary containing the embeddings of the speakers, this function finds an embedding of a given speaker_id using regular expression.

    import re
    speaker_mapping_dict = load_speaker_mapping(speaker_embedding_filepath)

    def find_matches(d, item):
        for k in d:
            name = str(d[k]['name'])
            if re.match('^'+item, name):
                return d[k]['embedding']

    speaker_re = f"{speaker_id}*"
    return find_matches(speaker_mapping_dict, speaker_re)
'''

def extract_reference_embedding(se_speaker_manager, extract_reference_embedding, use_cuda):
    
    reference_emb = se_speaker_manager.compute_embedding_from_clip(extract_reference_embedding)
    return reference_emb


def synthesize_waveform(model, config, sentence, reference_emb, speaker_name, use_cuda=True, language_name='pt-br', use_griffin_lim = False)-> np.array:
    '''
    Run inference on the model.
    '''
    speaker_id = model.speaker_manager.ids[speaker_name]
    language_id = model.language_manager.ids[language_name]
    waveform, alignment, _, _ = synthesis(
                                    model=model,
                                    text=sentence,
                                    CONFIG=config,
                                    use_cuda=use_cuda,
                                    speaker_id=speaker_id,
                                    style_wav=None,
                                    style_text=None,
                                    use_griffin_lim=use_griffin_lim,
                                    d_vector=reference_emb,
                                    language_id=language_id
                                ).values()
    return waveform


def generate_wavfile(model, config, se_speaker_manager, sentences, speaker_name, ref_wav_filepath, output_folder='output_inference', sr=24000, audio_format='wav', use_cuda=True, language_name='pt-br') -> None:
    '''
    Run inference on the model and save a wav file
    '''

    reference_emb = extract_reference_embedding(se_speaker_manager, ref_wav_filepath, use_cuda)

    for index, sentence in enumerate(tqdm(sentences)):

        waveform = synthesize_waveform(model, config, sentence, reference_emb, speaker_name, use_cuda, language_name)
        filename = f"output-{index}"
        filepath = os.path.join(output_folder, filename)
        save_file(filepath, waveform, audio_format, sr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--checkpoint', default='/home/fred/Projetos/YourTTS/yourtts-du_en_fr_ge_it_pl_ptbr_sp-August-07-2022_01+09AM-0cbaa0f6/checkpoint_930000.pth', help='Checkpoint pth filepath')
    parser.add_argument('--config', default='/home/fred/Projetos/YourTTS/yourtts-du_en_fr_ge_it_pl_ptbr_sp-August-07-2022_01+09AM-0cbaa0f6/config.json', help='Config json filepath')
    parser.add_argument('--checkpoint_se', default='/home/fred/Projetos/YourTTS/checkpoints/official/yourtts_official/model_se.pth', help='Speaker Encoder checkpoint pth filepath')
    parser.add_argument('--config_se', default='/home/fred/Projetos/YourTTS/checkpoints/official/yourtts_official/config_se.json', help='Speaker Encoder config json filepath')
    parser.add_argument('--speaker_embeddings', default='/home/fred/Projetos/YourTTS/yourtts-du_en_fr_ge_it_pl_ptbr_sp-August-07-2022_01+09AM-0cbaa0f6/d_vector_pl_pt.json', help='Speaker embeddings json filepath')
    parser.add_argument('--language_embeddings', default='/home/fred/Projetos/YourTTS/yourtts-du_en_fr_ge_it_pl_ptbr_sp-August-07-2022_01+09AM-0cbaa0f6/language_ids.json',
                        help='Speaker embeddings json filepath')
    parser.add_argument('--ref_wav', default='/home/fred/Artigo_MLS_dataset/result_norm_mls_portuguese_opus_revised-24k/dev/audio/5417/3702/5417_3702_000002-0001.wav',
                        help='Reference audiofile wav filepath')
    parser.add_argument('--speaker', default='1890\n', help="Speaker name")
    parser.add_argument('--language', default='pt-br', help="Language name")
    parser.add_argument('--input_file', default='sentences.txt', help='Sentences input file')
    parser.add_argument('--output_folder', default='output_inference', help='Output folder')
    parser.add_argument('--sr', default=24000)
    parser.add_argument('--audio_format', default='wav')
    parser.add_argument("--show_speakers", default=False, action="store_true")
    parser.add_argument("--show_languages", default=False, action="store_true")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    #DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, config = load_model_config(args.checkpoint, args.config, args.speaker_embeddings, args.language_embeddings)

    if args.show_speakers:
        print(model.speaker_manager.get_speakers())
        exit()
    if args.show_languages:
        print(model.language_manager.language_names)
        exit()
    with open(args.input_file) as f:
        sentences = f.readlines()

    se_speaker_manager = SpeakerManager(
                            encoder_model_path=args.checkpoint_se, 
                            encoder_config_path=args.config_se, 
                            use_cuda=use_cuda
                         )

    output_folder = os.path.join(args.base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    generate_wavfile(model, config, se_speaker_manager, sentences, args.speaker, args.ref_wav, output_folder, args.sr, 'wav', use_cuda, args.language)


if __name__ == "__main__":
    main()
