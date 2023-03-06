import requests
import pandas as pd
from tqdm import tqdm

from tasks.ukiyo_loader import UkiyoDataLoader, UkiyoDataOptions


def full_data():
    # read dataset
    df_all = pd.read_csv('../arc_meisho.csv')

    df_all = df_all.drop(df_all.columns[0], axis=1)

    print(df_all.loc[df_all['link'] == 'nan'])

    print(df_all['link'][0])

    # preprocessing
    df_all['title'] = df_all['title'].str.split('・').str[-1]
    #
    # # download images from urls
    # count = 0
    # list_unvailable = []
    # for url in tqdm(df_all['link']):
    #
    #     response = requests.get(url)
    #     if response.status_code == 200:  # check if url exists
    #         file = open(f"../../data/images/{count}.jpg", "wb")
    #         file.write(response.content)
    #         file.close()
    #     else:
    #         list_unvailable.append(count)
    #         #print('Web site does not exist')
    #         #print(url)
    #
    #     count+=1
    #
    # # def write_list_in_txt(path: str, txt_name: str, list_values :list):
    # # def read_list_in_txt(path: str, txt_name: str) -> list :
    #
    # # # open file in write mode
    # # with open(r'../../data/images/list_unvailable.txt', 'w') as fp:
    # #     for item in list_unvailable:
    # #         # write each item on a new line
    # #         fp.write("%s\n" % item)
    # #     print('Done')
    #


def download_images(list_urls: list):
    # download images from urls
    count = 0
    list_img_names = []
    img_availability = [True] * len(list_urls)
    for url in tqdm(list_urls):
        name_image = url.split('/')[-1]
        num_image = "%03d" % count  # eg '001', '002' : place a 0 in front of numbers if they are less than hundred

        response = requests.get(url)
        if response.status_code == 200:  # check if url exists
            file = open(f"../../data/images/{num_image}.jpg", "wb")
            file.write(response.content)
            file.close()
            list_img_names.append(f"{num_image}.jpg")
        else:
            img_availability[count] = False
            # print('Web site does not exist')
            # print(url)

        count += 1

    # def write_list_in_txt(path: str, txt_name: str, list_values :list):
    # def read_list_in_txt(path: str, txt_name: str) -> list :

    # # open file in write mode
    # with open(r'../../data/images/list_unvailable.txt', 'w') as fp:
    #     for item in list_unvailable:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('Done')

    return img_availability, list_img_names


import pandas as pd
from collections import defaultdict


def top_n_frequent_entities(df, entity_col, title_col, entity_label, n=200):
    # Create a defaultdict to count the frequency and positions of each entity
    entity_counts = defaultdict(lambda: {'count': 0, 'positions': []})

    # Iterate through the rows in the DataFrame and update the entity counts
    for i, row in df.iterrows():
        entity_positions = row[entity_col]
        title = row[title_col]
        for start, end, label in entity_positions:
            if label == entity_label:
                entity = title[start:end]
                entity_counts[entity]['count'] += 1
                entity_counts[entity]['positions'].append((start, end, title))

    # Sort the entity counts by the frequency in descending order
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1]['count'], reverse=True)

    # Return the top n most frequent entities
    top_n_entities = []
    for entity, counts in sorted_entities:
        if counts['count'] > 0:
            top_n_entities.append({'entity': entity, 'count': counts['count']})  # , 'positions': counts['positions']})
        if len(top_n_entities) == n:
            break

    return top_n_entities


def train_test_data():
    df_train = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TRAIN_TITLE,
                               data_path='../').loader()  # read dataset

    df_test = UkiyoDataLoader(type_of_dataset=UkiyoDataOptions.TEST_TITLE,
                              data_path='../').loader()

    print(df_train.columns)

    df = df_train.append(df_test, ignore_index=True)

    most_frequent = top_n_frequent_entities(df, 'entities', 'title', entity_label='PLACE')
    print(most_frequent)

    # List of substrings to search for
    top_places = {'富士': "Fuji",
                  '京都': 'Kyoto',
                  '東海道': 'Tokaido',
                  '江戸': 'Edo',
                  '広沢池': 'Hirosawa Pond'
                  }
    substrings = top_places.keys()
    # Regular expression pattern to match any of the substrings
    pattern = '|'.join(substrings)
    # Filter the DataFrame based on whether the "text" column includes at least one substring
    filtered_df = df[df['title'].str.contains(pattern)]
    # Add a new column with the detected substrings
    filtered_df['detected_substrings'] = filtered_df['title'].str.findall(pattern)

    img_availability, list_img_names = download_images(list_urls=filtered_df['Image URL'])
    filtered_df['img_availability'] = img_availability
    filtered_df['img_name'] = list_img_names

    print(filtered_df.head(85))

    filtered_df.to_csv('../../data/images/gold_dataset.csv', index=False)


if __name__ == "__main__":
    train_test_data()
