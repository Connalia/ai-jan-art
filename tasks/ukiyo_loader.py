__all__ = ["UkiyoDataLoader", "UkiyoDataset"]

from enum import auto
from strenum import StrEnum

from ast import literal_eval

import re

import pandas as pd


class UkiyoDataset(StrEnum):
    TRAIN_TITLE = auto()  # train_place.csv: train set with annonated tag PLACE
    TEST_TITLE = auto()  # test_place.csv: test set with annonated tag PLACE
    MEISHO = auto()  # arc_meisho.csv: whole dataset without annotations


class UkiyoDataLoader:

    def __init__(self, type_of_dataset: UkiyoDataset, data_path: str = '../../data/'):
        self.type_of_dataset = type_of_dataset
        self.data_path = data_path
        self.df_ukiyo = None

    @staticmethod
    def extract_title(df_ukiyo: pd.DataFrame) -> pd.DataFrame:

        # keep the original title of meisho dataset in column 'full_title'
        df_ukiyo['full_title'] = df_ukiyo['title']

        # keep only the title part after last token '・'
        df_ukiyo['title'] = df_ukiyo['title'].str.split('・').str[-1]

        ### Drop rows without titles ###
        # Drop column with nan : there are 51 title with ''
        df_ukiyo = df_ukiyo[df_ukiyo['title'] != '']

        # Drop column with only - : there are 11 title with ''
        df_ukiyo = df_ukiyo[df_ukiyo['title'] != '－']
        #################################

        # delete all the `〈NUMBER〉`
        df_ukiyo['title'] = df_ukiyo['title'].apply(lambda title: re.sub("〈[A-Za-z0-9_]+〉", "", title))

        return df_ukiyo

    def loader(self) -> pd.DataFrame:

        if self.type_of_dataset == UkiyoDataset.TRAIN_TITLE:

            self.df_ukiyo = pd.read_csv(self.data_path + 'train_place.csv')

            # column of csv as dtype list
            self.df_ukiyo['entities'] = self.df_ukiyo['entities'].apply(literal_eval)

        elif self.type_of_dataset == UkiyoDataset.TEST_TITLE:

            self.df_ukiyo = pd.read_csv(self.data_path + 'test_place.csv')

            # column of csv as dtype list
            self.df_ukiyo['entities'] = self.df_ukiyo['entities'].apply(literal_eval)

        elif self.type_of_dataset == UkiyoDataset.MEISHO:

            self.df_ukiyo = pd.read_csv(self.data_path + 'arc_meisho.csv')

            # drop first column that belong to index
            self.df_ukiyo = self.df_ukiyo.iloc[:, 1:]

            self.df_ukiyo = self.extract_title(self.df_ukiyo)

        return self.df_ukiyo


######################################################################

if __name__ == "__main__":
    df_ukiyo = UkiyoDataLoader(type_of_dataset=UkiyoDataset.MEISHO,
                               data_path='../data/').loader()
    df_ukiyo.to_csv('../data/arc_meisho_full.csv', index=False)
    print(df_ukiyo)
