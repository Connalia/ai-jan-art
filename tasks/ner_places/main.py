from enum import auto
from strenum import StrEnum

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDatasetOptions

from tasks.ner_places.gazetteer import ModelGazetteer
from tasks.ner_places.geolod import ModelGeolod


class NerModel(StrEnum):
    GAZETTEER = auto()
    GEOLOD = auto()
    BERT_FINETUNE = auto()
    SPACY = auto()


def main() -> None:

    model_run = [NerModel.GAZETTEER, NerModel.GEOLOD]

    df_test = UkiyoDataLoader(type_of_dataset=UkiyoDatasetOptions.TEST_TITLE).loader()

    if NerModel.GAZETTEER in model_run:
        ModelGazetteer(df_test).runner()
    if NerModel.GEOLOD in model_run:
        ModelGeolod(df_test).runner()

if __name__ == "__main__":
    main()
