- `info_location.csv`: include meta data, like Latitude and Longitude
  for places that extract from Bert model

- `arc_meisho.csv`: the whole dataset with ukiyo-e from Art Research Center's
  Ukiyo-e Portal Database

- `arc_meisho_full.csv`: the whole dataset with ukiyo-e from Art Research Center's
  Ukiyo-e Portal Database + with extra column without unsuseful metainfo

|                    title |                                   full_title |
|-------------------------:|---------------------------------------------:|
|     芳年「東海道　京都之内」「大内能上覧図」 |          arcUP0542文久０３・・芳年「東海道　京都之内」「大内能上覧図」 |
| 広景「江戸名所道戯盡」「廿九」「虎の御門外の景」 | 025-C003-030安政０６・10・広景「江戸名所道戯盡」「廿九」「虎の御門外の景」 |

|                                                                 actual_img_link Japanese |                                                                               thub_img_link | 
|-----------------------------------------------------------------------------------------:|--------------------------------------------------------------------------------------------:|
| https://www.arc.ritsumei.ac.jp/archive01/theater/image/PB/arc/Prints/arcUP/arcUP0542.jpg | https://www.arc.ritsumei.ac.jp/archive01/theater/th_image/PB/arc/Prints/arcUP/arcUP0542.jpg | 
|                                   https://data.ukiyo-e.org/metro/scaled/025-C003-030.jpg |                                      https://data.ukiyo-e.org/metro/scaled/025-C003-030.jpg |

- `jap_gazetteer.xlsx`: Gazetteer
  from [Geospatial Information Authority of Japan (GSI)](https://www.gsi.go.jp/ENGLISH/pape_e300284.html)
  (last updated 2007) . The dataset extracts
  from [gsi_japan_000042053.pdf](https://github.com/Connalia/ai-jan-art/blob/main/doc/!data/gsi_japan_000042053.pdf)

Sample:

| Japanese (Kanji) | Japanese (Hiragana) | Romanized Japanese | Longitude | Latitude | Classification |
|-----------------:|--------------------:|-------------------:|----------:|---------:|---------------:|
|              網走川 |              あばしりがわ |      Abashiri Gawa |  144゜ 14’ |  44゜ 00’ |          River |
|              網走湖 |               あばしりこ |        Abashiri Ko |  144゜ 10’ |  43゜ 58’ |           Lake |

    **NOTE** 

    other purchased Japan dataset: https://simplemaps.com/data/jp-cities

- `train_place.csv`: the annotated train ukiyo-e dataset with tag PLACE (last updated 2022/02/22)

  `test_place.csv`: the annotated test ukiyo-e dataset with tag PLACE (last updated 2022/02/22)

  Note: The datasets extract from data/annotated_title.py based on Ewa & Marita

|               title |                             entities |          Genre | Artist | 
|--------------------:|-------------------------------------:|---------------:|-------:|
| 「東海道五十三次」 「三十八」「藤川」 | [(1, 4, 'PLACE'), (17, 19, 'PLACE... | 名所絵  五十三次  美人画 |     北斎 |  
|     「東都六玉顔ノ内」 「角田川」 | [(1, 3, 'PLACE'), (12, 15, 'PLACE... |       美人画  名所絵 |     国貞 |   

|                                                                            Image URL |                                          Permalink |
|-------------------------------------------------------------------------------------:|---------------------------------------------------:|
| https://www.arc.ritsumei.ac.jp/archive01/theater/image/PB/PVT/Ebi/Prints/Ebi0043.jpg | https://www.dh-jac.net/db/nishikie/Ebi0043/2021d7/ |
| https://www.arc.ritsumei.ac.jp/archive01/theater/image/PB/PVT/Ebi/Prints/Ebi0091.jpg | https://www.dh-jac.net/db/nishikie/Ebi0091/2021d7/ |
