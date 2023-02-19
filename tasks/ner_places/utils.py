__all__ = ['change_string_to_int_tags',
           'extract_tags_per_char']

import pandas as pd
from tqdm import tqdm

from src.logs import *


def change_string_to_int_tags(label_series: pd.Series) -> pd.Series:
    '''
    Change string Tags to binary int Tags
    :return:
    label_series : eg self.df_test['per_char_tag']
    '''

    for i in range(len(label_series)):
        label_series[i] = [item.replace('PLACE', '1') for item in label_series[i]]
        label_series[i] = [item.replace('O', '0') for item in label_series[i]]

        # using list comprehension to perform conversion str to in
        label_series[i] = [int(i) for i in label_series[i]]

    return label_series


def extract_tags_per_char(df) -> pd.DataFrame:
    extended_logger.extra_info("######### extract_tags_per_char ######### ")

    list_tags = []
    list_pos_tags = []

    noTag = 'O'

    for i in range(len(df['title'])):

        text = df['title'][i]  # eg.'朝食にを焼いて食べまし[MASK]。'
        tags = list(df['entities'][i])

        extended_logger.extra_info(f'Title: {text}')
        extended_logger.extra_info(f'Tags: {tags}')
        extended_logger.extra_info(f'Number of Tags: {len(tags)}')

        list_in = [noTag] * len(text)  # initialize   # eg.['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        for tag in tags:  # each title has multiple tags
            begin = tag[0]
            end = tag[1]

            text_pos = text[begin:end]
            name_tag = tag[2]

            list_in[begin:end] = [name_tag] * len(
                text_pos)  # replace the position per char with Tag eg. ['O', 'O', 'O', 'O', .....] -> (2, 5, 'PLACE') ['O', 'O', 'PLACE', 'PLACE', 'PLACE', 'O', 'O', 'O']

            extended_logger.extra_info(f'Token: {text_pos} -> Tag: {name_tag}')

        extended_logger.extra_info(str(list_in))
        list_tags.append(list_in)

    df["per_char_tag"] = list_tags  # update dataframe with tag per character

    # #covert column word_labels from list to string
    # #eg [O, O, PLACE, PLACE, O, O, O, O, O, O, O, O, O, O, O] -> 'O,O,PLACE,PLACE,O,O,O,O,O,O,O,O,O,O,O'
    # for i in range(len(data)):
    #     data['word_labels'][i] = ",".join(data['word_labels'][i])

    return df
