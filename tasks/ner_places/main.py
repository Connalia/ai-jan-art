from enum import auto
from strenum import StrEnum

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDataOptions

from tasks.ner_places.gazetteer import ModelGazetteer
from tasks.ner_places.geolod import ModelGeolod

from src.logs import *

'''
Fine-tuning means taking some machine learning model that 
has already learned something before (i.e. been trained on some data) 
and then training that model (i.e. training it some more, possibly on different data).
 
 VS
 
Transfer learning means to apply the knowledge that some machine learning model holds 
(represented by its learned parameters) to a new (but in some way related) task.
'''


class NerModel(StrEnum):
    GAZETTEER = auto()
    GEOLOD = auto()
    BERT_NER = auto()
    FINE_TUNE_BERT_NER = auto()
    SPACY = auto()


def main(models: list) -> None:
    df_test = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TEST_TITLE).loader()

    if NerModel.GAZETTEER in models:
        ModelGazetteer(df_test).runner()
        extended_logger.success(f"End GAZETTEER")
        extended_logger.critical(f"Not Implement Evalutation yet")  # ToDo
    if NerModel.GEOLOD in models:
        ModelGeolod(df_test).runner()
        extended_logger.success(f"End Geolod")
    if NerModel.BERT_NER in models:
        extended_logger.critical(f"Not Implement yet")  # ToDo
    if NerModel.FINE_TUNE_BERT_NER in models:
        extended_logger.critical(f"Not Implement yet")  # ToDo


if __name__ == "__main__":
    model_run = [  # NerModel.GAZETTEER,
        NerModel.GEOLOD,
        NerModel.FINE_TUNE_BERT_NER]
    main(models=model_run)
