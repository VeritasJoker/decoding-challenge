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
    model = whisper.load_model(model_size)

    return model


def load_whisper_model_by_path(model_path, checkpoint):
    processor = WhisperProcessor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path)

    model_path = os.path.join(model_path, f"checkpoint-{checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        output_hidden_states=True,
        return_dict=True,
    )

    return model, processor, tokenizer


# def hf_whisper(tokens, device):  # hf whisper
#     model = WhisperForConditionalGeneration.from_pretrained(
#         "openai/whisper-tiny.en"
#     ).to(device)
#     input = torch.randn(1, 80, 3000).to(device)
#     output = model(input_features=input, decoder_input_ids=tokens)
#     return output


# def git_whisper(tokens, device):  # whisper
#     model = whisper.load_model("tiny.en").to(device)
#     input = torch.randn(1, 80, 3000).to(device)
#     output = model(input, tokens)
#     return output


# def ecog_whisper(tokens, device):  # conv + decoder
#     model = EcogModel(ModelDimensions()).to(device)  # conv + decoder
#     input = torch.randn(1, 256, 500).to(device)
#     output = model(input, tokens)
#     # output = model.forward(input, tokens)
#     return output


def main():
    args = parse_arguments()

    breakpoint()

    device = "cpu"
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    tokens = tokenizer.encode("hey what's up", return_tensors="pt").to(device)
    model = EcogModel(ModelDimensions()).to(device)  # conv + decoder
    breakpoint()
    return


if __name__ == "__main__":
    main()
