import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import whisper
import whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)

from model_modules import EcogModel, ModelDimensions
from model_config import parse_arguments


def load_whisper_model_by_hf(model_size):
    """Loading whisper model from HuggingFace
    Args:
        model_size: whisper model type

    Return:
        model: Whisper model
        processor: Whisper processor
        tokenizer: Whisper tokenizer
    """
    model_fullname = f"openai/whisper-{model_size}"
    print(f"Loading {model_fullname}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_fullname,
        output_hidden_states=True,
        return_dict=True,
    )
    processor = WhisperProcessor.from_pretrained(model_fullname)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_fullname, add_prefix_space=True, predict_timestamps=True
    )

    return model, processor, tokenizer


def load_whisper_model_by_git(model_size):
    """Loading whisper model from HuggingFace
    Args:
        model_size: whisper model type

    Return:
        model: Whisper model
        None
        None
    """
    model = whisper.load_model(model_size)

    return model, None, None


def load_whisper_model_by_path(model_path, checkpoint):
    """Loading whisper model by path
    Args:
        model_path: path of the model
        checkpoint: checkpoint step

    Return:
        model: Whisper model
        processor: Whisper processor
        tokenizer: Whisper tokenizer
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path)

    model_path = os.path.join(model_path, f"checkpoint-{checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        output_hidden_states=True,
        return_dict=True,
    )

    return model, processor, tokenizer


def main():
    args = parse_arguments()
    breakpoint()

    _, processor, tokenizer = load_whisper_model_by_hf(args.model_size)
    model = EcogModel(ModelDimensions(n_feature_dim=args.feature_dim)).to(args.device)

    breakpoint()
    # print("Saving processor")
    # processor.save_pretrained(args.saving_dir)

    return


if __name__ == "__main__":
    main()
