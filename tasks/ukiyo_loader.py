__all__ = ["UkiyoDataLoader", "UkiyoDataOptions"]

from src.utils.print_custom import printBeautyTable
from src.utils.options import Options

from ast import literal_eval

import re

import pandas as pd

from enum import auto
from strenum import StrEnum


# TRAIN_TITLE = auto() not work yet with auto

class UkiyoDataOptions(StrEnum):
    """ Available Ukiyo-e Datasets"""

    # Sorted Class Variable
    TRAIN_TITLE = auto()  # train_place.csv: train set with annonated tag PLACE
    TEST_TITLE = auto()  # test_place.csv: test set with annonated tag PLACE
    MEISHO = auto()  # arc_meisho_old.csv: whole meisho dataset from ARC db
    FULL_MEISHO = auto()  # arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info
    MEISHO_OLD = auto()  # arc_meisho_old.csv: whole dataset without annotations

    @staticmethod
    def meta_info():
        """
        Print to explain of what is each variable

        :return: Print the follow

        Information for variables:
        +-------------+--------------------------------------------------------------------------------------+
        |   Variable  |                                         Name                                         |
        +-------------+--------------------------------------------------------------------------------------+
        |    MEISHO   | arc_meisho_old.csv: whole dataset without annot. + preprocessing to be as  FULL_MEISHO   |
        | FULL_MEISHO | arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info |
        |  TEST_TITLE |                  test_place.csv: test set with annonated tag PLACE                   |
        | TRAIN_TITLE |                 train_place.csv: train set with annonated tag PLACE                  |
        +-------------+--------------------------------------------------------------------------------------+
        """
        # take a list of all class variables
        list_class_var = [member.name for member in UkiyoDataOptions]

        names = ['train_place.csv: train set with annonated tag PLACE',
                 'test_place.csv: test set with annonated tag PLACE',
                 'arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info',
                 'arc_meisho_old.csv: whole dataset without annotations',
                 ]
        print('Information for variables:')
        table = [list(zipped) for zipped in
                 zip(list_class_var, names)]  # combine list
        printBeautyTable(table)


# class UkiyoDataOptions(Options):
#     """ Available Ukiyo-e Datasets"""
#
#     # Sorted Class Variable
#     TRAIN_TITLE = 'TRAIN_TITLE'  # train_place.csv: train set with annonated tag PLACE
#     TEST_TITLE = 'TEST_TITLE'  # test_place.csv: test set with annonated tag PLACE
#     FULL_MEISHO = 'FULL_MEISHO'  # arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info
#     MEISHO = 'MEISHO'  # arc_meisho_old.csv: whole dataset without annotations
#
#     @staticmethod
#     def meta_info():
#         """
#         Print to explain of what is each variable
#
#         :return: Print the follow
#
#         Information for variables:
#         +-------------+--------------------------------------------------------------------------------------+
#         |   Variable  |                                         Name                                         |
#         +-------------+--------------------------------------------------------------------------------------+
#         |    MEISHO   |                  arc_meisho_old.csv: whole dataset without annotations                   |
#         | FULL_MEISHO | arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info |
#         |  TEST_TITLE |                  test_place.csv: test set with annonated tag PLACE                   |
#         | TRAIN_TITLE |                 train_place.csv: train set with annonated tag PLACE                  |
#         +-------------+--------------------------------------------------------------------------------------+
#         """
#
#         names = ['train_place.csv: train set with annonated tag PLACE',
#                  'test_place.csv: test set with annonated tag PLACE',
#                  'arc_meisho_full_old.csv: whole dataset without annot. + title without unuseful meta info'
#                  'arc_meisho_old.csv: whole dataset without annotations',
#                  ]
#         print('Information for variables:')
#         table = [list(zipped) for zipped in
#                  zip(UkiyoDataOptions.get_string_options(), sorted(names))]  # combine list
#         printBeautyTable(table)


class UkiyoDataLoader:

    def __init__(self, type_of_dataset: UkiyoDataOptions = UkiyoDataOptions.FULL_MEISHO,
                 data_path: str = '../../data/'):
        self.type_of_dataset = type_of_dataset
        self.data_path = data_path
        self.df_ukiyo = None

    @staticmethod
    def extract_title(df_ukiyo: pd.DataFrame) -> pd.DataFrame:
        """ Extract the main title without unuseful metadata and
            add it in 'title' column and keep the old title in 'full_title' column
            E.g.
            From: arcUP0542文久０３・・芳年「東海道　京都之内」「大内能上覧図」
            To: 芳年「東海道　京都之内」「大内能上覧図」
        """

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
        """
        Select and load the available datasets
        """

        if self.type_of_dataset == UkiyoDataOptions.TRAIN_TITLE:

            self.df_ukiyo = pd.read_csv(self.data_path + 'train_place.csv')

            # column of csv as dtype list
            self.df_ukiyo['entities'] = self.df_ukiyo['entities'].apply(literal_eval)

        elif self.type_of_dataset == UkiyoDataOptions.TEST_TITLE:

            self.df_ukiyo = pd.read_csv(self.data_path + 'test_place.csv')

            # column of csv as dtype list
            self.df_ukiyo['entities'] = self.df_ukiyo['entities'].apply(literal_eval)

        elif self.type_of_dataset == UkiyoDataOptions.MEISHO:

            self.df_ukiyo = pd.read_csv(self.data_path + 'arc_meisho.csv')

        elif self.type_of_dataset == UkiyoDataOptions.MEISHO_OLD:

            print('Warning: Depricated Dataset!!!')

            self.df_ukiyo = pd.read_csv(self.data_path + 'arc_meisho_old.csv')

            # drop first column that belong to index
            self.df_ukiyo = self.df_ukiyo.iloc[:, 1:]

            self.df_ukiyo = self.extract_title(self.df_ukiyo)

            # add urls of images
            self.df_ukiyo.rename(columns={"link": "thub_img_link"}, inplace=True)
            self.df_ukiyo["actual_img_link"] = self.df_ukiyo["thub_img_link"].str.replace('th_image', 'image')

        elif self.type_of_dataset == UkiyoDataOptions.FULL_MEISHO:

            print('Warning: Depricated Dataset!!!')

            self.df_ukiyo = pd.read_csv(self.data_path + 'arc_meisho_full_old.csv')

        else:  # user select not exist dataset
            while True:
                print(UkiyoDataOptions.meta_info())
                dataset_valid_check_name = input("Please select valid dataset:")
                if dataset_valid_check_name in list(UkiyoDataOptions):
                    self.type_of_dataset = dataset_valid_check_name
                    self.loader()
                    break

        return self.df_ukiyo


######################################################################

if __name__ == "__main__":
    df_ukiyo = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.MEISHO,
                               data_path='../data/').loader()
    df_ukiyo.to_csv('../data/arc_meisho_full_old.csv', index=False)
    print(df_ukiyo)
