import argparse
import os
import sys
TTS_PATH = "TTS/"
# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

import time

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

# model vars 
#MODEL_PATH = 'checkpoints_vits_multispeaker_ptbr/best_model.pth.tar'
#CONFIG_PATH = 'checkpoints_vits_multispeaker_ptbr/config.json'
#MODEL_PATH = './checkpoints/vits_tts_mls-April-13-2022_07+46PM-2d64d351/best_model.pth.tar'
#CONFIG_PATH = './checkpoints/vits_tts_mls-April-13-2022_07+46PM-2d64d351/config.json'

MODEL_PATH = './checkpoints/vits_tts_mls-May-21-2022_09+23PM-2d64d351/checkpoint_1260000.pth.tar'
CONFIG_PATH = './checkpoints/vits_tts_mls-May-21-2022_09+23PM-2d64d351/config.json'


USE_CUDA = torch.cuda.is_available()
sr = 22050

speaker_idx = {
    "AS": 0,
    "C1": 1,
    "C2": 2,
    "DI": 3,
    "GM": 4,
    "JA": 5,
    "JC": 6,
    "LI": 7,
    "LR": 8,
    "PE": 9,
    "PM": 10,
    "RG": 11,
    "VC": 12    
}

# load the config
C = load_config(CONFIG_PATH)

# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

#C.model_args['d_vector_file'] = TTS_SPEAKERS

model = setup_model(C)
#model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
# print(model.language_manager.num_languages, model.embedded_language_dim)
# print(model.emb_l)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(cp['model'])

model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

model.noise_scale = 0.0  # defines the noise variance applied to the random z vector at inference.
model.length_scale = 1.0  # scaler for the duration predictor. The larger it is, the slower the speech.
model.noise_scale_w = 1.0 # defines the noise variance applied to the duration predictor z vector at inference.

model.inference_noise_scale = 0.667 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.0 # defines the noise variance applied to the duration predictor z vector at inference.

######################################################
# Save wav or ogg output file
######################################################
def save_file(filename, audio, output_type='wav', sampling_rate=22050):

    if output_type == 'wav':
        sf.write(filename + '.wav', audio, sampling_rate, 'PCM_16')

    else:
        # Ocorreu um bug da lib soundfile: corta o final do arquivo ogg.
        sf.write(filename + '.ogg', audio, sampling_rate, format='ogg', subtype='vorbis')

# A simpler version of what run.py does. It returns the created model and its saved checkpoint
def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None)
    return model, checkpoint

def load_tacotron_gst(logdir, scope):
    #carrega tacotron-gst
    args_T2S = ["--config_file=config/text2speech/tacotron_gst.py",
            "--mode=interactive_infer",
            "--logdir=" + logdir,
            "--batch_size_per_gpu=" + str(batch_size),
    ]
    
    model_T2S, checkpoint_T2S = get_model(args_T2S, scope)
    
    # Create the session and load the checkpoints
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)
    vars_T2S = {}
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        if scope in v.name:
            vars_T2S["/".join(v.op.name.split("/")[1:])] = v
    saver_T2S = tf.train.Saver(vars_T2S)
    saver_T2S.restore(sess, checkpoint_T2S)
    
    print('tacotron carregada..................')

    return model_T2S, sess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Adiciona pausa ao final do áudio passado
def add_pausa(full_audio, pausa, pausa_padrao, sample_rate):
    pausa = int(pausa_padrao * sample_rate) if pausa == -1 else int(pausa * sample_rate/1000) 
    full_audio = np.append(full_audio, np.zeros(pausa, dtype=np.int16))
    return full_audio

def generate(texts, voz, key, pausa_segundos=0.5, audio_format = 'wav'):

    print(texts)
    start = time.time()
    # Defining the speaker
    speaker_id = speaker_idx[voz]

    full_audio = np.empty((0,1), dtype=np.int16) 

    for i, TEXT in enumerate(texts):

        #texto, pausa = TEXT
        texto = TEXT
        pausa = 0.1
        print(" > Text: {}".format(texto))
        #speaker_id = i % 10
        wav, _, _, _ = synthesis(
                        model,
                        texto,
                        C,
                        "cuda" in str(next(model.parameters()).device),
                        ap,
                        speaker_id=speaker_id,
                        d_vector=None,
                        style_wav=None,
                        enable_eos_bos_chars=C.enable_eos_bos_chars,
                        do_trim_silence=False,
                    ).values()

        full_audio = np.append(full_audio, wav)

        if i + 1 != len(texts):
            full_audio = add_pausa(full_audio, pausa, pausa_segundos, sr)

    path = f'./temp_wavs/{key}'
    save_file(path, full_audio, audio_format, sr)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--speaker', default='C1', help='Speaker code')
    parser.add_argument('--input_file', default='texto_copel.txt', help='Sentences input file')
    parser.add_argument('--filename', default='output', help='Output filename')
    parser.add_argument('--output_folder', default='temp_wavs', help='Output folder')
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = f.readlines()

    output_folder = os.path.join(args.base_dir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for index, sentence in enumerate(lines):
        filename = args.filename + str(index)
        filepath = os.path.join(output_folder, filename)
        sentence = [sentence]
        generate(sentence, args.speaker, filename, float(0.5))


if __name__ == "__main__":
    main()


