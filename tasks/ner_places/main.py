from enum import auto
from strenum import StrEnum

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDataOptions

from tasks.ner_places.gazetteer import ModelGazetteer
from tasks.ner_places.geolod import ModelGeolod

from src.models.unsupervised.bert_retrain import FurtherBERT

from src.models.supervised.bert_ner import NerBERT

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
    FURTHER_BERT_NER = auto()
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
        # --> BERT Transfer learning to NER with labeled Ukiyo-e Titles
        df_test = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TEST_TITLE).loader()
        df_train = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TRAIN_TITLE).loader()
        NerBERT(Train=Train, Test=Test, checkpoint='cl-tohoku/bert-base-japanese',
                tokinizer_name='cl-tohoku/bert-base-japanese').runner()

    if NerModel.FURTHER_BERT_NER in models:
        # --> BERT Fine-tune to Mask Language Task with All Ukiyo-e Titles
        df_ukiyo = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.FULL_MEISHO,
                                   data_path='data/').loader()

        FurtherBERT(checkpoint="cl-tohoku/bert-base-japanese",
                    df=df_ukiyo).runner()

        # --> BERT Transfer learning to NER with labeled Ukiyo-e Titles
        df_test = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TEST_TITLE).loader()
        df_train = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TRAIN_TITLE).loader()
        NerBERT(Train=Train, Test=Test, checkpoint='bert-japanese-finetuned-meisho',
                tokinizer_name='cl-tohoku/bert-base-japanese').runner()


if __name__ == "__main__":
    model_run = [  # NerModel.GAZETTEER,
        NerModel.GEOLOD,
        NerModel.FURTHER_BERT_NER]
    main(models=model_run)
