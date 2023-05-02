__all__ = ['NerBERT']

import pandas as pd
from transformers import AutoTokenizer
from transformers import BertForTokenClassification

from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

from transformers import TrainingArguments
from transformers import Trainer

from termcolor import colored

from sklearn.metrics import classification_report, accuracy_score

import numpy as np

from tqdm import tqdm

from src.logs import *

import math

SEED = 42

import torch

from torch.utils.data import Dataset, DataLoader

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
extended_logger.extra_info(device)


def test_convert_dataframe_tolistTuple():  # Not use in project
    # initialize list of lists
    data = [['「東海道　京都名所之内」「四条河原」', [(1, 4, "PLACE"), (5, 7, "PLACE"), (13, 17, "PLACE")]],
            ['「東海道　京都名所之内」「四条河原」', [(1, 4, "PLACE"), (5, 7, "PLACE"), (13, 17, "PLACE")]]
            ]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['title', 'entities'])

    # print dataframe.
    extended_logger.dev_info(df.head(3))

    list = [("「東海道　京都名所之内」「四条河原」", {"entities": [(1, 4, "PLACE"), (5, 7, "PLACE"), (13, 17, "PLACE")]}),
            ("「東海道名所之内」「御能拝見之図」", {"entities": [(1, 4, "PLACE")]})]

    for entity in list:
        text = entity[0]
        tags = entity[1]['entities']
        extended_logger.dev_info(text)
        extended_logger.dev_info(tags)
        extended_logger.dev_info('__________')

    extended_logger.dev_info([tuple(x) for x in df.to_numpy()])


class DatasetBERT(Dataset):
    def __init__(self, dataframe, tokenizer, MAX_LEN, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index]  # .strip().split()
        word_labels = self.data.word_labels[index].split(",")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  #  is_pretokenized=True,
                                  #  return_offsets_mapping=True,
                                  padding='max_length',
                                  #  truncation=True,
                                  max_length=self.MAX_LEN)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.labels_to_ids[label] for label in word_labels]

        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        # encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_labels = np.ones(self.MAX_LEN, dtype=int) * -100

        # # set labels whose first offset position is 0 and the second is not 0
        # i = 0
        # for idx, mapping in enumerate(encoding["offset_mapping"]):
        #   if mapping[0] == 0 and mapping[1] != 0:
        #     # overwrite label
        #     encoded_labels[idx] = labels[i]
        #     i += 1

        # set labels
        for idx in range(len(labels)):
            # overwrite label
            encoded_labels[idx] = labels[idx]

        # # step 4: turn everything into PyTorch tensors
        # item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len


class NerBERT():
    """
    Fine-tuning BERT for named-entity recognition

    For more info: see tutorial in notebooks\tutorials\Custom_Named_Entity_Recognition_with_BERT.ipynb
    = https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=MyETdB-dkBsX
    """

    # TODO unit test
    # TODO hyperparameter tuning
    def __init__(self,  # sentence_colname: str = "title",
                 # dataset_path: str = None, df: pd.DataFrame = None,
                 format_data: str = "dataframe_pos",
                 Train: list = None, Test: list = None,
                 list_tags: list = ['O', 'PLACE'],
                 checkpoint: str = 'cl-tohoku/bert-base-japanese',  # 'bert-japanese-finetuned-meisho',
                 tokinizer_name: str = 'cl-tohoku/bert-base-japanese',
                 MAX_LEN: int = 60,
                 TRAIN_BATCH_SIZE: int = 4,
                 VALID_BATCH_SIZE: int = 2,
                 EPOCHS: int = 20,
                 LEARNING_RATE: float = 1e-05,
                 MAX_GRAD_NORM: int = 10,
                 *args, **kwargs):

        """
        :param format_data: format TRAIn and Test data

                --> Input : format_data : "list_pos" = a list with pair of sentence and spacy tag's dictionary
                                                        as extract from spacy.
                Train = [("「東海道　京都名所之内」「四条河原」", {"entities":[(1,4,"PLACE"),(5,7,"PLACE"),(13,17,"PLACE")]}),
                         ("「東海道名所之内」「御能拝見之図」", {"entities":[(1,4,"PLACE")]} ),...,]

                --> Input : format_data : "dataframe_pos" = a dataframe with position of spacy tag's .
                                    title	        entities
                0	「東海道　京都之内」「大内能上覧図」	[(1, 4, PLACE), (5, 7, PLACE)]
                1	「東海道　京都名所之内」「四条河原」	[(1, 4, PLACE), (5, 7, PLACE), (13, 17, PLACE)]
        """
        self.format_data = format_data
        self.Train = Train
        self.Test = Test

        # Hyperparameters
        self.MAX_LEN = MAX_LEN
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.VALID_BATCH_SIZE = VALID_BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.MAX_GRAD_NORM = MAX_GRAD_NORM

        # User add extra Hyperparameters
        defaults_model_hyperparam = {}  # {'random_state': SEED}
        self.updated_values = {**defaults_model_hyperparam, **kwargs}  # overwrite kwargs over default values
        extended_logger.dev_info(f"args: {args}")
        extended_logger.dev_info(f"kwargs: {self.updated_values}")

        # Pretrain Models Name
        self.labels_to_ids = {k: v for v, k in enumerate(list_tags)}  # {'O': 0, 'PLACE': 1}
        self.ids_to_labels = {v: k for v, k in enumerate(list_tags)}  # {0: 'O', 1: 'PLACE'}
        self.tokenizer = AutoTokenizer.from_pretrained(tokinizer_name)

        self.checkpoint = checkpoint
        self.model = None

        # Initilize
        self.training_loader = None
        self.testing_loader = None

    def extract_tags(self, data: list):
        """
        :param data: data with a specific format:
                --> Input : format_data : "list_pos" = a list with pair of sentence and spacy tag's dictionary
                                                        as extract from spacy.
                Train = [("「東海道　京都名所之内」「四条河原」", {"entities":[(1,4,"PLACE"),(5,7,"PLACE"),(13,17,"PLACE")]}),
                         ("「東海道名所之内」「御能拝見之図」", {"entities":[(1,4,"PLACE")]} ),...,]

                --> Input : format_data : "dataframe_pos" = a dataframe with position of spacy tag's .
                                    title	        entities
                0	「東海道　京都之内」「大内能上覧図」	[(1, 4, PLACE), (5, 7, PLACE)]
                1	「東海道　京都名所之内」「四条河原」	[(1, 4, PLACE), (5, 7, PLACE), (13, 17, PLACE)]

        :return: A new column with NER tags/labels as list per sentence
                --> Output
                                    sentence	word_labels
                「東海道　京都名所之内」「四条河原」	O,O,PLACE,PLACE,O,O,O,O,O,PLACE,PLACE,O,O
                「東海道名所之内」「御能拝見之図」	O,O,PLACE,O,O,O,O,O,O,O,O,O,O,O,O,O

        --> Output Print:
        ------------------------------------
        Title: 「東海道　京都名所之内」「四条河原」
        Title Encode: ['[CLS]', '「', '東海道', '京都', '名所', '之', '内', '」', '「', '四条', '河原', '」', '[SEP]']
        Tags: [(1, 4, 'PLACE'), (5, 7, 'PLACE'), (13, 17, 'PLACE')]
        Number of Tags: 3
        Token: 東海道 -> Tag: PLACE
        [2]
        Token: 京都 -> Tag: PLACE
        [3]
        Token: 四条河原 -> Tag: PLACE
        [9, 10]
        ------------------------------------

        Title: 「東海道名所之内」「御能拝見之図」
        Title Encode: ['[CLS]', '「', '東海道', '名所', '之', '内', '」', '「', '御', '能', '拝', '##見', '之', '図', '」', '[SEP]']
        Tags: [(1, 4, 'PLACE')]
        Number of Tags: 1
        Token: 東海道 -> Tag: PLACE
        [2]
        ------------------------------------
        """
        list_title = []
        list_token_title = []
        list_encode_title = []
        list_tags = []

        noTag = 'O'

        if self.format_data == "dataframe_pos":  # convert to list_pos format
            data = [tuple(x) for x in data.to_numpy()]  # convert dataframe to array of tuples

        for entity in tqdm(data):
            ### Title ###
            text = entity[0]

            print('\nTitle:', text)

            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            print('Title Encode:', tokens)

            list_title.append(text)  # eg.'朝食にを焼いて食べまし[MASK]。'
            list_token_title.append(
                tokens)  # eg.['[CLS]', '朝食', 'に', 'を', '焼い', 'て', '食べ', 'まし', '[MASK]', '。', '[SEP]']
            list_encode_title.append(token_ids)  # eg.[2, 25965, 7, 11, 16878, 16, 2949, 3913, 4, 8, 3]

            if self.format_data == "dataframe_pos":
                tags = entity[1]
            else:
                tags = entity[1]['entities']

            print('Tags:', tags)
            print('Number of Tags:', len(tags))

            list_in = [noTag] * len(tokens)  # eg.['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

            # each title has multiple tags
            for tag in tags:
                begin = tag[0]
                end = tag[1]

                text_token = text[begin:end]
                name_tag = tag[2]

                print(f'Token: {text_token} -> Tag: {name_tag}')

                #########################################################

                # Finding all indexes of a string in the list
                # We want:
                # either text_token include in token  eg. '朝食' = '朝食'
                # or token include in text_token 祇園大鳥居 = '祇園' '大' '鳥居'

                # TODO !!!προσοχή δεν εχουμε φτιαξει ακομα την περιπτωση 祇園大鳥居 = '祇園' '大' '##鳥', μεχρι στιγμεις θα βαλει 'PLACE' 'PLACE' '0'!!!
                # TODO !!!προσοχή δεν εχουμε φτιαξει ακομα την περιπτωση 栂尾門= '[UNK]', '門前', μεχρι στιγμεις θα βαλει '0' '0'!!!

                indices = [i for i, s in enumerate(tokens) if (text_token in s) or (s in text_token)]
                print(indices)

                # add the tag in the correct token
                for ind in indices:
                    list_in[ind] = name_tag
                #########################################################

            list_tags.append(list_in)
            print('------------------------------------')

            # intialise data of lists.

        data = {'title': list_title,
                'title_token': list_token_title,
                'title_encode': list_encode_title,
                'tags': list_tags}

        # Create DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Create DataFrame
        data = pd.DataFrame({'sentence': list_title, 'word_labels': list_tags})

        # covert column word_labels from list to string
        # eg [O, O, PLACE, PLACE, O, O, O, O, O, O, O, O, O, O, O] -> 'O,O,PLACE,PLACE,O,O,O,O,O,O,O,O,O,O,O'
        for i in range(len(data)):
            data['word_labels'][i] = ",".join(data['word_labels'][i])

        print(data)

        return data

    def data_loader(self):
        # From position number start/end tags To tag per token
        train_dataset = self.extract_tags(self.Train)
        test_dataset = self.extract_tags(self.Test)

        # Tranform to dataset hugging Face
        # print("FULL Dataset: {}".format(data.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))
        """ Tranform to dataset hugging Face
                --> From train_dataset.head(0)
                                    sentence	word_labels
                「東海道　京都之内」「大内能上覧図」	O,O,PLACE,PLACE,O,O,O,O,O,O,O,O,O,O,O

                --> To training_set[0]
                {'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 'input_ids': tensor([    2,    36,  7174,  1316,  3376,   186,    38,    36, 10576,  1329,
                           109, 29643,   903,    38,     3,     0,     0,     0,     0,     0,
                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]),
                 'labels': tensor([   0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,
                            0,    0,    0, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]),
                 'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
                """
        training_set = DatasetBERT(train_dataset, self.tokenizer, self.MAX_LEN, self.labels_to_ids)
        testing_set = DatasetBERT(test_dataset, self.tokenizer, self.MAX_LEN, self.labels_to_ids)

        # for token, label in zip(self.tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]),
        #                         training_set[0]["labels"]):
        #     print('{0:10}  {1}'.format(token, label))
        """


            --> Output print    
            [CLS]       0
            「           0
            東海道         1
            京都          1
            之           0
            内           0
            」           0
            「           0
            大内          0
            能           0
            上           0
            ##覧         0
            図           0
            」           0
            [SEP]       0
            [PAD]       -100
            [PAD]       -100
            [PAD]       -100
            [PAD]       -100
            [PAD]       -100
            [PAD]       -100
            [PAD]       -100
        """

        train_params = {'batch_size': self.TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        test_params = {'batch_size': self.VALID_BATCH_SIZE,
                       'shuffle': True,
                       'num_workers': 0
                       }

        self.training_loader = DataLoader(training_set, **train_params)
        self.testing_loader = DataLoader(testing_set, **test_params)

        return training_set

    def runner(self):

        training_set = self.data_loader()

        # Fine tune BERT Models
        # TODO check difference from BertForSequenceClassification
        self.model = BertForTokenClassification.from_pretrained(self.checkpoint,
                                                                num_labels=len(self.labels_to_ids),
                                                                return_dict=False)
        # model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=3)
        self.model.to(device)

        inputs = training_set[2]
        input_ids = inputs["input_ids"].unsqueeze(0)
        attention_mask = inputs["attention_mask"].unsqueeze(0)
        labels = inputs["labels"].unsqueeze(0)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        initial_loss = outputs[0]

        tr_logits = outputs[1]

        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.LEARNING_RATE)

        for epoch in tqdm(range(self.EPOCHS)):
            print(f"\nTraining epoch: {epoch + 1}")
            self.trainner(optimizer, epoch)

        # evaluation metrics
        labels, predictions = self.valid(self.model, self.testing_loader)
        print(classification_report(labels, predictions))

    # Defining the training function on the 80% of the dataset for tuning the bert model
    def trainner(self, optimizer, epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.model.train()

        for idx, batch in enumerate(self.training_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            loss, tr_logits = self.model(input_ids=ids,
                                         attention_mask=mask,
                                         labels=labels)
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
            # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.MAX_GRAD_NORM
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def valid(self, model, testing_loader):
        # put model in evaluation mode
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(testing_loader):

                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.long)

                loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)

                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += labels.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

                # compute evaluation accuracy
                flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
                active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)

                eval_labels.extend(labels)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy

        labels = [self.ids_to_labels[id.item()] for id in eval_labels]
        predictions = [self.ids_to_labels[id.item()] for id in eval_preds]

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Validation Loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")

        return labels, predictions

    def predict_sentence(self, sentence):

        inputs = self.tokenizer(sentence,
                                padding='max_length',
                                max_length=self.MAX_LEN,
                                return_tensors="pt")
        # item = {key: torch.as_tensor(val) for key, val in encoding.items()}

        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        # forward pass
        outputs = self.model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits,
                                             axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        prediction_all = []
        # set predicted labels
        for token_pred in range(len(wp_preds)):

            if wp_preds[token_pred][0] == '[CLS]' or wp_preds[token_pred][0] == '[SEP]' or wp_preds[token_pred][
                0] == '[PAD]':
                continue
            elif wp_preds[token_pred][1] == 'O':
                prediction_all.append(wp_preds[token_pred])
            else:
                # predictions
                prediction.append(wp_preds[token_pred])
                prediction_all.append(wp_preds[token_pred])
                # print(wp_preds[token_pred][0])

        return prediction, prediction_all

    def inference(self, df_titles: pd.Series):
        titles = []
        tags_pred = []
        pos_pred = []

        count = 0
        for i in tqdm(range(len(df_titles))):
            count += 1

            # # ERROR: from indexing
            # if count==28 or count==73 or count==76 or count==138:
            #     continue

            sentence = df_titles[i]  # eg "「東海道　京都之内」「大内能上覧図」"
            prediction, prediction_all = self.predict_sentence(sentence)

            print('Title:', count, colored((sentence), 'red', attrs=['bold']))
            print('Predictions:', prediction)

            # doc2 = nlp.make_doc(sentence)

            # spans = []
            # for pred in prediction: # eg pred = ('東海道', 'LOC')
            #     #Find start and end positions of all occurrences within a string in Python
            #     text = pred[0] # eg pred[0] = '東海道'
            #     text = text.replace('##\\','「')
            #     text = text.replace('##や','「')
            #     text = text.replace('[','「')
            #     for match in re.finditer(text, sentence):
            #         temp = [match.start(), match.end(),pred[1]] # eg pred[1] = 'LOC'
            #         spans.append(temp)

            # print(spans)

            titles.append(sentence)
            tags_pred.append(prediction)
            # pos_pred.append(spans)

        # intialise data of lists.
        data_silver = {'Title': titles,
                       'Tags': tags_pred}

        # Calling DataFrame constructor on list
        df_silver = pd.DataFrame(data_silver)
        print(df_silver)

        df_silver.to_csv('silver.csv', index=False)


if __name__ == "__main__":
    test_dataframe_tolistTuple()
