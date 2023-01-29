__all__ = ['FurtherTrainBERT']

import pandas as pd
from transformers import AutoTokenizer

from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

from transformers import TrainingArguments
from transformers import Trainer

from src.logs import *

import math

SEED = 42


class FurtherTrainBERT:
    """
    Mask Language Model further train of Hugging Face pretrain BERT model

    For more info: see tutorial in notebooks/bert_further_pretrain.ipynb
    """

    # TODO add kwarg** for more hyperparameters
    # TODO unit test
    # TODO hyperparameter tuning
    def __init__(self, checkpoint: str,
                 dataset_path: str = None, df: pd.DataFrame = None,
                 sentence_colname: str = "title",
                 model_name: str = 'bert-japanese-finetuned-meisho',
                 chunk_size: int = 128, batch_size: int = 64,
                 split_size: float = 0.2,
                 *args, **kwargs):
        """
        :param checkpoint: model name of pretrain BERT of Hugging Face
        :param dataset_path: dataset path that load the dataset through hugging face format
        :param sentence_colname: the column name of dataset that include the sentence for mask laguage task
        :param chunk_size: note: pick something a bit smaller from tokenizer.model_max_length (512) that can fit in memory
        :param model_name: the name of the folder that save the fine-tune model after training
        :param split_size: percentage size of test set
        """

        self.checkpoint = checkpoint
        self.dataset_path = dataset_path
        self.df = df

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForMaskedLM.from_pretrained(self.checkpoint)

        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.sentence_colname = sentence_colname
        self.model_name = model_name

        defaults_model_hyperparam = {'random_state': SEED}
        self.updated_values = {**defaults_model_hyperparam, **kwargs}  # overwrite kwargs over default values
        extend_logging.meta_info(f"args: {args}")
        extend_logging.meta_info(f"kwargs: {self.updated_values}")

    def tokenize_function(self, examples):
        result = self.tokenizer(examples[self.sentence_colname])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    @staticmethod
    def group_texts(examples, chunk_size: int = 128):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # Create a new labels column
        """ Create a new labels column
        masked language modeling the objective is to predict randomly masked tokens
        in the input batch, and by creating a labels column we provide the ground truth
        for our language model to learn from.
        """
        result["labels"] = result["input_ids"].copy()

        return result

    def loader(self):
        """ Load data with Hugging Face format
        For example:DatasetDict({
                        unsupervised: Dataset({
                            features: ['title', 'link', 'full_title'],
                            num_rows: 20346
                        })
                    })

        :return: DatasetDict = Hugging Face Dataset Dict
        """
        if self.dataset_path is not None:
            from datasets import load_dataset
            data_files = {"unsupervised": self.dataset_path}
            dataset_dict = load_dataset("csv", data_files=data_files)

            # dataset = dataset_dict["unsupervised"]

        elif self.df is not None:  # Pandas DataFrame to Hugging Face
            from datasets import Dataset
            dataset = Dataset.from_pandas(self.df)

            '''
                        Create a Dataset Dict Format
                        From
                        Dataset({
                            features: ['title', 'link', 'full_title'],
                            num_rows: 20346
                        })
                        To
                        DatasetDict({
                            unsupervised: Dataset({
                                features: ['title', 'link', 'full_title'],
                                num_rows: 20346
                            })
                        })
                        '''
            from datasets import DatasetDict
            dataset_dict = DatasetDict()
            dataset_dict['unsupervised'] = dataset

        else:  # user select not exist dataframe or path
            extend_logging.error('Not valid dataframe or data path on BERT Mask Language Model')
            exit()

        return dataset_dict

    def runner(self):

        # LOAD DATASET -------------------------------------------------------------------------------------------------
        meisho_dataset = self.loader()

        # PREPROCESSING ------------------------------------------------------------------------------------------------

        meisho_features = meisho_dataset["unsupervised"].features.keys()  # eg dict_keys(['title', 'link'])

        # tokinize text
        tokenized_datasets = meisho_dataset.map(
            self.tokenize_function,
            batched=True,  # use batched=True to activate fast multithreading!
            remove_columns=meisho_features  # remove all the init column as we need only tokizised sentece
        )

        # split into chunck
        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True)

        # split dataset
        sample_size = lm_datasets["unsupervised"].num_rows
        train_size = int((1 - self.split_size) * sample_size)  # train_size = 10
        test_size = int(self.split_size * sample_size)  # test_size = int(0.1 * self.train_size)

        downsampled_dataset = lm_datasets["unsupervised"].train_test_split(
            train_size=train_size, test_size=test_size, seed=SEED
        )

        # MODEL --------------------------------------------------------------------------------------------------------

        # add [MASK] on labels
        """
        Inserting [MASK] tokens at random positions in the inputs using `DataCollatorForLanguageModeling`

        In `DataCollatorForLanguageModeling`, the `mlm_probability` argument that specifies what fraction of the 
        tokens to mask. We’ll pick 15%, which is the amount used for BERT and a common choice in the literature:"""
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=0.15)

        # Including logging_steps to ensure we track the training loss with each epoch
        # Show the training loss with every epoch
        logging_steps = len(downsampled_dataset["train"]) // self.batch_size
        if logging_steps <= 0:
            logging_steps = 1

        training_args = TrainingArguments(
            output_dir=self.model_name,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=True,  # to enable mixed-precision training, which gives us another boost in speed
            logging_steps=logging_steps,  # logging_steps, #to ensure we track the training loss with each epoch
            **self.updated_values
            # push_to_hub=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=downsampled_dataset["train"],
            eval_dataset=downsampled_dataset["test"],
            data_collator=data_collator,
        )

        # compute the resulting perplexity on the test set before finetune
        eval_results_before = trainer.evaluate()  # compute the cross-entropy loss on the test set

        # Fine-tune the model (training mode)
        trainer.train()

        # compute the resulting perplexity on the test set after finetune
        eval_results = trainer.evaluate()
        print(f">>> Perplexity before further pretrain BERT: {math.exp(eval_results_before['eval_loss']):.2f}")
        print(f">>> Perplexity after further pretrain BERT: {math.exp(eval_results['eval_loss']):.2f}")

        return


######################################################################

if __name__ == "__main__":
    FurtherTrainBERT(checkpoint="cl-tohoku/bert-base-japanese",
                     dataset_path=".../.../.../arc_meisho_full.csv").runner()
