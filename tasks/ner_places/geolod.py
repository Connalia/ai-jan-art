__all__ = ['ModelGeolod']

"""" NER PLACES using GeoLOD

! Run first the notebook: ner_place_geolod.ipynb
! Save results: results/ner_places/geolod_annotations_place.csv
"""

import pandas as pd
from tqdm import tqdm
import itertools

from ast import literal_eval

from sklearn.metrics import classification_report

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDatasetOptions
from tasks.ner_places.utils import extract_tags_per_char, change_string_to_int_tags


class ModelGeolod():

    def __init__(self, df_test):
        self.df_test = df_test

    def inference(self):
        print("######### Inference of GeoLOD ######### ")

    def evaluator(self, actuals, predictions):
        print('##################### GeoLOD #####################')
        print(classification_report(actuals, predictions))

    def runner(self):
        df_prediction = pd.read_csv('../../results/ner_places/df_geolod.csv')
        # column of csv as dtype list
        df_prediction['labels_predict'] = df_prediction['labels_predict'].apply(literal_eval)

        self.df_test = extract_tags_per_char(self.df_test)

        # change string Tags to binary int Tags
        labels_actual_series = change_string_to_int_tags(label_series=self.df_test['per_char_tag'])
        self.df_test['labels_actual'] = labels_actual_series

        # join list of lists in python
        actuals = list(itertools.chain.from_iterable(self.df_test['labels_actual']))
        predictions = list(itertools.chain.from_iterable(df_prediction['labels_predict']))

        self.evaluator(actuals, predictions)


######################################################################

if __name__ == "__main__":
    df_test = UkiyoDataLoader(type_of_dataset=UkiyoDatasetOptions.TEST_TITLE).loader()
    ModelGeolod(df_test).runner()
