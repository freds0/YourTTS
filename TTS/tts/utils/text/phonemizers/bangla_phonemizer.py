from typing import Dict

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

# Lazy import to avoid Python 3.9 compatibility issues with bangla library
_bangla_text_to_phonemes = None

def _get_bangla_phonemizer():
    global _bangla_text_to_phonemes
    if _bangla_text_to_phonemes is None:
        try:
            from TTS.tts.utils.text.bangla.phonemizer import bangla_text_to_phonemes
            _bangla_text_to_phonemes = bangla_text_to_phonemes
        except (ImportError, TypeError) as e:
            raise ImportError(
                "Bangla phonemizer requires the 'bangla' library, but it's not compatible with Python 3.9. "
                "Please use Python 3.10+ or install a compatible version of the 'bangla' library."
            ) from e
    return _bangla_text_to_phonemes

_DEF_ZH_PUNCS = "、.,[]()?!〽~『』「」【】"


class BN_Phonemizer(BasePhonemizer):
    """🐸TTS bn phonemizer using functions in `TTS.tts.utils.text.bangla.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_ZH_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.

    Example ::

        "这是，样本中文。" -> `d|ʒ|ø|4| |ʂ|ʏ|4| |，| |i|ɑ|ŋ|4|b|œ|n|3| |d|ʒ|o|ŋ|1|w|œ|n|2| |。`

    TODO: someone with Bangla knowledge should check this implementation
    """

    language = "bn"

    def __init__(self, punctuations=_DEF_ZH_PUNCS, keep_puncs=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "bn_phonemizer"

    @staticmethod
    def phonemize_bn(text: str, separator: str = "|") -> str:  # pylint: disable=unused-argument
        bangla_text_to_phonemes = _get_bangla_phonemizer()
        ph = bangla_text_to_phonemes(text)
        return ph

    def _phonemize(self, text, separator):
        return self.phonemize_bn(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"bn": "Bangla"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
    txt = "রাসূলুল্লাহ সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম শিক্ষা দিয়েছেন যে, কেউ যদি কোন খারাপ কিছুর সম্মুখীন হয়, তখনও যেন বলে."
    e = BN_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print("`" + e.phonemize(txt) + "`")
