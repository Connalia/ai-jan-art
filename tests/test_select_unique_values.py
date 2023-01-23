from enum import auto
from strenum import StrEnum


class NerModel(StrEnum):
    GAZETTEER = auto()
    GEOLOD = auto()
    BERT = auto()
    SPACY = auto()


assert NerModel.GAZETTEER == "GAZETTEER"  # correct

assert NerModel.GAZETTEER == "gazetteer"  # wrong
