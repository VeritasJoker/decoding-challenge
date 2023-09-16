import os
import torch
import pickle
import numpy as np
import pandas as pd
import whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)


def load_whisper_model_hf(model_size):
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


def transcribe_audio(model, processor, filename):
    # load and prepare audio
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    input_features = mel.unsqueeze(dim=0)

    # model generate (greedy decoding)
    output = model.generate(inputs=input_features, max_new_tokens=448)
    transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

    return transcription


def transcribe_spec(model, processor, spec):
    # prepare spec
    spec = torch.from_numpy(spec)
    input_features = spec.unsqueeze(dim=0)

    # model generate (greedy decoding)
    output = model.generate(inputs=input_features, max_new_tokens=448)
    transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

    return transcription
