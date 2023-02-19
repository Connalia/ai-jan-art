import requests
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    #read dataset
    df_all = pd.read_csv('../../data/arc_meisho.csv')

    df_all = df_all.drop(df_all.columns[0], axis=1)

    print(df_all.loc[df_all['link']=='nan'])

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

