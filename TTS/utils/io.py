import os
import pickle as pickle_tts
from typing import Any, Callable, Dict, Union

import fsspec
import torch

from TTS.utils.generic_utils import get_user_data_dir


class RenamingUnpickler(pickle_tts.Unpickler):
    """Overload default pickler to solve module renaming problem"""

    def find_class(self, module, name):
        return super().find_class(module.replace("mozilla_voice_tts", "TTS"), name)


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)


def load_checkpoint(
    model, checkpoint_path, use_cuda=False, eval=False, cache=False
):  # pylint: disable=redefined-builtin
    """Load model checkpoint with backward compatibility.

    Supports loading from:
    1. Old checkpoints with 'model' key (original format)
    2. PyTorch Lightning checkpoints with 'state_dict' key

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        use_cuda: Whether to move model to CUDA
        eval: Whether to set model to eval mode
        cache: Whether to cache the checkpoint file

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    try:
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), pickle_module=pickle_tts, cache=cache)

    # Handle different checkpoint formats
    if "model" in state:
        # Old format with 'model' key
        model.load_state_dict(state["model"])
    elif "state_dict" in state:
        # PyTorch Lightning format
        # Filter out Lightning-specific keys (e.g., 'vits_model.xxx' -> 'xxx')
        model_state_dict = {}
        for key, value in state["state_dict"].items():
            if key.startswith("vits_model."):
                # Remove 'vits_model.' prefix for compatibility
                new_key = key[len("vits_model."):]
                model_state_dict[new_key] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Try loading directly (maybe it's just a state_dict)
        model.load_state_dict(state)

    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state
