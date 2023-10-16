import os
from scipy.io import loadmat
import pickle
import pandas as pd
import numpy as np
import torch
import string
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict

from data_main import load_data
from model_build import load_ecog_model_by_whisper
from model_config import parse_arguments


def model_tokenize_word(labels, tokenizer):
    labels = pd.DataFrame(labels)
    labels = labels.rename(columns={0: "label"})
    # labels["label_lower"] = labels.label.str.lower()  # lower case
    labels["token_ids"] = labels.label.astype(str).apply(tokenizer.encode)  # tokenize

    # get rid of punctuation tokens
    # puncs = []
    # for punc in string.punctuation:  # 1-by-1 (weird interactions if not)
    #     puncs = puncs + tokenizer.encode(punc)[2:-1]  # tokenize punctuation
    # labels["token_ids_nopunc"] = labels.token_ids.apply(
    #     lambda x: [token for token in x if token not in puncs]
    # )

    # get rid of empty tokens
    labels["token_ids_nospace"] = labels.token_ids.apply(
        lambda x: [token for token in x if token != 220]
    )

    return labels.token_ids_nospace.tolist()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def remove_punc(input_str):
    for character in string.punctuation:  # remove punc
        input_str = input_str.replace(character, "")
    return input_str.lower()  # lowercase


def get_first_word(sentence):
    if len(sentence.split()) > 0:
        first_word = sentence.split()[0]  # get first word
        return first_word
    else:
        return sentence  # empty prediction


def compute_metric(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # pred_str = [remove_punc(pred) for pred in pred_str]
    # label_str = [remove_punc(label) for label in label_str]
    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    pred_str = [get_first_word(pred) for pred in pred_str]
    label_str = [get_first_word(label) for label in label_str]
    first_cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer, "first_cer": first_cer}


def get_trainer(args, data_all):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.saving_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=3000,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=3,
        save_strategy="no",
        save_steps=500,
        eval_steps=100,
        logging_steps=25,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_all["train"],
        eval_dataset=data_all["test"],
        data_collator=data_collator,
        compute_metrics=compute_metric,
        tokenizer=processor.tokenizer,
    )

    return trainer, training_args


def prepare_data(args):
    """Load training and testing data
    Args:
        args (Namespace):  commandline arguments

    Return:
        data_all (DatasetDict): dataset for train and test
    """
    # load data
    train_x, train_y = load_data(args, "train")
    test_x, test_y = load_data(args, "test")

    # tokenize labels
    train_y = model_tokenize_word(train_y, tokenizer)
    test_y = model_tokenize_word(test_y, tokenizer)

    ecog_data_train = {
        "input_features": train_x,
        "labels": train_y,
    }
    ecog_data_test = {
        "input_features": test_x,
        "labels": test_y,
    }
    ecog_data_train = Dataset.from_dict(ecog_data_train)
    ecog_data_test = Dataset.from_dict(ecog_data_test)
    data_all = DatasetDict({"train": ecog_data_train, "test": ecog_data_test})
    return data_all


def main():
    # Read command line arguments
    args = parse_arguments()

    print("Init model / metric")  # Initialize model and metric
    global model, processor, tokenizer, metric_cer, metric_wer, seg_type  # global variables
    model, processor, tokenizer = load_ecog_model_by_whisper(args)
    metric_cer = evaluate.load("./metrics/cer")
    metric_wer = evaluate.load("./metrics/wer")

    print("Prepare data")  # Prepare Data
    data_all = prepare_data(args)

    print("Start training")  # Training
    trainer, _ = get_trainer(args, data_all)
    trainer.train()

    return None


if __name__ == "__main__":
    main()
