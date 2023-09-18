import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import whisper
import argparse
from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import pickle


def load_pickle(filepath):
    print(f"Loading: {filepath}")
    with open(filepath, "rb") as fh:
        pkl = pickle.load(fh)
    return pkl


# def test_hf_whisper(tokens, device):  # hf whisper
#     model = WhisperForConditionalGeneration.from_pretrained(
#         "openai/whisper-tiny.en"
#     ).to(device)
#     input = torch.randn(1, 80, 3000).to(device)
#     output = model(input_features=input, decoder_input_ids=tokens)
#     return output


# def test_git_whisper(tokens, device):  # whisper
#     model = whisper.load_model("tiny.en").to(device)
#     input = torch.randn(1, 80, 3000).to(device)
#     output = model(input, tokens)
#     return output


# def test_ecog_whisper(tokens, device):  # conv + decoder
#     model = EcogModel(ModelDimensions()).to(device)  # conv + decoder
#     input = torch.randn(1, 256, 500).to(device)
#     output = model(input, tokens)
#     # output = model.forward(input, tokens)
#     return output


def test_model(args, model, tokenizer):
    # Testing
    tokens = tokenizer.encode("hey what's up", return_tensors="pt").to(args.device)
    input = torch.randn(1, 128 * args.feature_dim, 500).to(args.device)
    output = model(input, tokens)
    return


def main():
    filepath = "../whisper-decoder/seg-data/podcast/chunk/717_ecog_stg_spec.pkl"
    data = load_pickle(filepath)
    breakpoint()

    return


if __name__ == "__main__":
    main()
