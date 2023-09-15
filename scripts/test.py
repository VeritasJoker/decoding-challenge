import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import whisper
import argparse
from model_build import EcogModel, ModelDimensions
from transformers import WhisperTokenizer, WhisperForConditionalGeneration


def hf_whisper(tokens, device):  # hf whisper
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en"
    ).to(device)
    input = torch.randn(1, 80, 3000).to(device)
    output = model(input_features=input, decoder_input_ids=tokens)
    return output


def git_whisper(tokens, device):  # whisper
    model = whisper.load_model("tiny.en").to(device)
    input = torch.randn(1, 80, 3000).to(device)
    output = model(input, tokens)
    return output


def ecog_whisper(tokens, device):  # conv + decoder
    model = EcogModel(ModelDimensions()).to(device)  # conv + decoder
    input = torch.randn(1, 256, 500).to(device)
    output = model(input, tokens)
    # output = model.forward(input, tokens)
    return output


def main():
    device = "cpu"
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    tokens = tokenizer.encode("hey what's up", return_tensors="pt").to(device)

    breakpoint()
    return


if __name__ == "__main__":
    main()
