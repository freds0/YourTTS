import argparse
import os
from argparse import RawTextHelpFormatter

from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser(
    description="""Compute embedding vectors for each wav file in a dataset.\n\n"""
    """
    Example runs:
    python TTS/bin/compute_embeddings.py speaker_encoder_model.pth speaker_encoder_config.json  dataset_config.json embeddings_output_path/
    """,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
parser.add_argument(
    "config_path",
    type=str,
    help="Path to model config file.",
)

parser.add_argument(
    "config_dataset_path",
    type=str,
    help="Path to dataset config file.",
)
parser.add_argument("output_path", type=str, help="path for output speakers.json and/or speakers.npy.")
parser.add_argument(
    "--old_file", type=str, help="Previous speakers.json file, only compute for new audios.", default=None
)
parser.add_argument("--use_cuda", type=bool, help="flag to set cuda. Default False", default=False)
parser.add_argument("--no_eval", type=bool, help="Do not compute eval?. Default False", default=False)

args = parser.parse_args()

c_dataset = load_config(args.config_dataset_path)

meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not args.no_eval)

if meta_data_eval is None:
    wav_files = meta_data_train
else:
    wav_files = meta_data_train + meta_data_eval

encoder_manager = SpeakerManager(
    encoder_model_path=args.model_path,
    encoder_config_path=args.config_path,
    d_vectors_file_path=args.old_file,
    use_cuda=args.use_cuda,
)

class_name_key = encoder_manager.encoder_config.class_name_key

# compute speaker embeddings
speaker_mapping = {}
for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, dict):
        class_name = wav_file[class_name_key]
        wav_file = wav_file["audio_file"]
    else:
        class_name = None

    wav_file_name = os.path.basename(wav_file)
    if args.old_file is not None and wav_file_name in encoder_manager.clip_ids:
        # get the embedding from the old file
        embedd = encoder_manager.get_embedding_by_clip(wav_file_name)
    else:
        # extract the embedding
        embedd = encoder_manager.compute_embedding_from_clip(wav_file)

    # create speaker_mapping if target dataset is defined
    speaker_mapping[wav_file_name] = {}
    speaker_mapping[wav_file_name]["name"] = class_name
    speaker_mapping[wav_file_name]["embedding"] = embedd

if speaker_mapping:
    # save speaker_mapping if target dataset is defined
    if ".json" not in args.output_path:
        mapping_file_path = os.path.join(args.output_path, "speakers.json")
    else:
        mapping_file_path = args.output_path

    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    # pylint: disable=W0212
    encoder_manager._save_json(mapping_file_path, speaker_mapping)
    print("Speaker embeddings saved at:", mapping_file_path)
