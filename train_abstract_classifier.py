import pandas
import glob
import logging
import os

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import load_from_disk
from datasets import DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('httpx').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

ABSTRACTS_COL = 'Abstract'
ABSTRACTS_DIR = 'data/scopus_2024_05_28'
ROOT_DIR = 'abstract_classifier_llm'
MODEL_PATH = os.path.join(ROOT_DIR, 'model')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINING_SET_PATH = os.path.join(DATA_DIR, 'train')
VALIDATION_SET_PATH = os.path.join(DATA_DIR, 'validation')
HOLDOUT_SET_PATH = os.path.join(DATA_DIR, 'holdout')


def load_abstracts(abstracts_dir):
    # List all CSV files in the directory
    csv_files = [
        file_path
        for file_path in glob.glob(os.path.join(abstracts_dir, '*.csv'))]

    # Load each file and select the "Abstracts" column
    dataframes = []
    for file_path in csv_files:
        LOGGER.info(f'loading {file_path}')
        df = pandas.read_csv(file_path, usecols=[ABSTRACTS_COL])
        dataframes.append(df)

    combined_dataframe = pandas.concat(dataframes, ignore_index=True)
    combined_dataframe = combined_dataframe.drop_duplicates()
    return combined_dataframe[ABSTRACTS_COL]


def main():
    abstract_series = load_abstracts(ABSTRACTS_DIR)

    while True:
        if not os.path.exists(MODEL_PATH):
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=2)
            os.makedirs(MODEL_PATH)
        else:
            tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

        if os.path.exists(TRAINING_SET_PATH):
            dataset = DatasetDict({
                'train': load_from_disk(TRAINING_SET_PATH),
                'validation': load_from_disk(VALIDATION_SET_PATH),
                'holdout': load_from_disk(HOLDOUT_SET_PATH),
            })

            def _tokenize_function(examples):
                return tokenizer(
                    examples[ABSTRACTS_COL],
                    padding="max_length", truncation=True)
            tokenized_datasets = dataset.map(_tokenize_function, batched=True)
            training_args = TrainingArguments(
                output_dir='./results',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_index_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
            )

            trainer.train()
            holdout_results = trainer.evaluate(tokenized_datasets['holdout'])
            print(holdout_results)
            save_model = False
            if save_model:
                model.save_pretrained(MODEL_PATH)
                tokenizer.save_pretrained(MODEL_PATH)
        else:
            dataset = DatasetDict({
                'train': {ABSTRACTS_COL: [], 'Labels': []},
                'validation': {ABSTRACTS_COL: [], 'Labels': []},
                'holdout': {ABSTRACTS_COL: [], 'Labels': []},
            })
        for abstract in abstract_series:
            print(abstract)
        break


if __name__ == '__main__':
    main()
