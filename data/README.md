- `info_location.csv`: include meta data, like Latitude and Longitude 
for places that extract from  Bert model

- `arc_meisho.csv`: the whole dataset with ukiyo-e from Art Research Center's
Ukiyo-e Portal Database

- `arc_meisho_full.csv`: the whole dataset with ukiyo-e from Art Research Center's
Ukiyo-e Portal Database + with extra column without unsuseful metainfo

- `jap_gazetteer.xlsx`: Gazetteer from [Geospatial Information Authority of Japan (GSI)](https://www.gsi.go.jp/ENGLISH/pape_e300284.html)
(last updated 2007) . The dataset extracts from [gsi_japan_000042053.pdf](https://github.com/Connalia/ai-jan-art/blob/main/doc/!data/gsi_japan_000042053.pdf)

Sample:
| Japanese (Kanji) | Japanese (Hiragana) | Romanized Japanese | Longitude | Latitude | Classification |
|------------------|---------------------|--------------------|-----------|----------|----------------|
| 網走川           | あばしりがわ        | Abashiri Gawa      | 144゜ 14’ | 44゜ 00’ | River          |
| 網走湖           | あばしりこ          | Abashiri Ko        | 144゜ 10’ | 43゜ 58’ | Lake           |

    **NOTE** 

    other purchased Japan dataset: https://simplemaps.com/data/jp-cities


- `train_place.csv`: the annotated train ukiyo-e dataset with tag PLACE (last updated 2022/02/22)

    `test_place.csv`: the annotated test ukiyo-e dataset with tag PLACE (last updated 2022/02/22)

    Note: The datasets extract from data/annotated_title.py based on Ewa & Marita

|                                 title |                                          entities |
|--------------------------------------:|--------------------------------------------------:|
| 「東海道五十三次」 「三十八」「藤川」 | [(1, 4, 'PLACE'), (17, 19, 'PLACE... |
|         「東都六玉顔ノ内」 「角田川」 | [(1, 3, 'PLACE'), (12, 15, 'PLACE... |
