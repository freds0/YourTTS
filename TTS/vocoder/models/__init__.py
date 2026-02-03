import importlib
import re

from coqpit import Coqpit


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(config: Coqpit):
    """Load models directly from configuration."""
    if "discriminator_model" in config and "generator_model" in config:
        MyModel = importlib.import_module("TTS.vocoder.models.gan")
        MyModel = getattr(MyModel, "GAN")
    else:
        MyModel = importlib.import_module("TTS.vocoder.models." + config.model.lower())
        if config.model.lower() == "gan":
            MyModel = getattr(MyModel, "GAN")
        else:
            try:
                MyModel = getattr(MyModel, to_camel(config.model))
            except ModuleNotFoundError as e:
                raise ValueError(f"Model {config.model} not exist!") from e
    print(" > Vocoder Model: {}".format(config.model))
    return MyModel.init_from_config(config)


def setup_generator(c):
    """TODO: use config object as arguments"""
    print(" > Generator Model: {}".format(c.generator_model))
    MyModel = importlib.import_module("TTS.vocoder.models." + c.generator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.generator_model))
    if c.generator_model.lower() in "hifigan_generator":
        model = MyModel(in_channels=c.audio["num_mels"], out_channels=1, **c.generator_model_params)
    else:
        raise NotImplementedError(f"Model {c.generator_model} not implemented!")
    return model


def setup_discriminator(c):
    """TODO: use config objekt as arguments"""
    print(" > Discriminator Model: {}".format(c.discriminator_model))
    MyModel = importlib.import_module("TTS.vocoder.models." + c.discriminator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.discriminator_model.lower()))
    if c.discriminator_model in "hifigan_discriminator":
        model = MyModel()
    elif c.discriminator_model in "random_window_discriminator":
        model = MyModel(
            cond_channels=c.audio["num_mels"],
            hop_length=c.audio["hop_length"],
            uncond_disc_donwsample_factors=c.discriminator_model_params["uncond_disc_donwsample_factors"],
            cond_disc_downsample_factors=c.discriminator_model_params["cond_disc_downsample_factors"],
            cond_disc_out_channels=c.discriminator_model_params["cond_disc_out_channels"],
            window_sizes=c.discriminator_model_params["window_sizes"],
        )
    else:
        raise NotImplementedError(f"Discriminator {c.discriminator_model} not implemented!")
    return model
