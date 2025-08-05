"""Track author affiliation by topic."""
import argparse
import random
import glob

import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState


MIN_LENGTH = 0
ID2LABEL = {
    0: 'university',
    1: 'research',
    2: 'practice',
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MODEL_PATH = 'saved_model'


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback based on the change in accuracy."""

    def __init__(self, accuracy_threshold=0.01, patience=1):
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.best_accuracy = None
        self.wait = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current_accuracy = kwargs['metrics']['eval_accuracy']
        if self.best_accuracy is None or current_accuracy > self.best_accuracy + self.accuracy_threshold:
            self.best_accuracy = current_accuracy
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"No improvement in accuracy for {self.patience} evaluations. Stopping training.")
                control.should_training_stop = True


def compute_metrics(accuracy):
    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    return _compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Train affiliation model')
    parser.add_argument(
        'affiliation_classifications', help='Path to affiliation list')
    args = parser.parse_args()

    training_set = {
        'test': [],
        'train': [],
    }

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    for affiliation_file in glob.glob(args.affiliation_classifications):
        print(affiliation_file)
        with open(affiliation_file, 'r', encoding='utf-8') as file:
            for line in file:
                classification = line.split(':')[0]
                affiliation = ':'.join(line.split(':')[1:])
                tokenized_affiliation = tokenizer(affiliation, truncation=True)
                entry = {
                    'label': LABEL2ID[classification],
                    'text': affiliation
                }
                entry.update(tokenized_affiliation)
                if random.random() > .2:
                    training_set['train'].append(entry)
                else:
                    training_set['test'].append(entry)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID)

    training_args = TrainingArguments(
        output_dir="affiliation_classifier",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set["train"],
        eval_dataset=training_set["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(accuracy),
        callbacks=[EarlyStoppingCallback(accuracy_threshold=0.01, patience=2)],
    )

    trainer.train()

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)


if __name__ == '__main__':
    main()