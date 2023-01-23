__all__ = ['ModelGazetteer']

"""" NER PLACES using Gazetteer 
The gazetteer was retrieved from https://www.gsi.go.jp/ENGLISH/pape_e300284.html
"""

import pandas as pd
from tqdm import tqdm

import itertools
from sklearn.metrics import classification_report

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDatasetOptions
from tasks.ner_places.utils import extract_tags_per_char, change_string_to_int_tags


class ModelGazetteer():

    def __init__(self, df_test):
        self.df_test = df_test
        self.df_gaz = None

    def inference(self):

        print("######### Inference of Gazetteer ######### ")

        list_predict = []

        for title in tqdm(self.df_test['title']):

            print('Title:', title)

            list_bool = [0] * len(title)
            position = []

            for place in self.df_gaz['Japanese (Kanji)']:  # self.df_gaz['Japanese (Hiragana)']

                if place in title:
                    print('Japanese Place (Kanji):', place)

                    start_pos = title.find(place)
                    end_pos = start_pos + len(place)

                    position.append((start_pos, end_pos))

                    list_bool[start_pos:end_pos] = [1] * len(place)

            print(list_bool)
            list_predict.append(list_bool)
        ######### DataFrame return #########

        # Create DataFrame
        df_predict = pd.DataFrame({'title': self.df_test['title'],
                                   'entities': self.df_test['entities'],
                                   'per_char_tag': self.df_test['per_char_tag'],
                                   'labels_predict': list_predict,
                                   })

        return df_predict

    def evaluator(self, actuals, predictions):
        print('##################### Gazetteer #####################')
        print(classification_report(actuals, predictions))

    def runner(self):

        # Load dataset
        self.df_gaz = pd.read_excel('../../data/jap_gazetteer.xlsx')

        self.df_test = extract_tags_per_char(self.df_test)

        df_predict = self.inference()

        # change string Tags to binary int Tags
        labels_actual_series = change_string_to_int_tags(label_series=df_predict['per_char_tag'])
        df_predict['labels_actual'] = labels_actual_series

        # join list of lists in python
        actuals = list(itertools.chain.from_iterable(df_predict['labels_actual']))
        predictions = list(itertools.chain.from_iterable(df_predict['labels_predict']))

        self.evaluator(actuals, predictions)


######################################################################

if __name__ == "__main__":
    df_test = UkiyoDataLoader(type_of_dataset=UkiyoDatasetOptions.TEST_TITLE).loader()
    ModelGazetteer(df_test).runner()
