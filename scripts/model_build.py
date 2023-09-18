import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from torch import Tensor, nn
import whisper
import whisper
from transformers import (
    WhisperConfig,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperModel,
    WhisperForConditionalGeneration,
)

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


def load_ecog_model_by_whisper(args):
    whisper_model, processor, tokenizer = load_whisper_model_by_hf(args.model_size)
    whisper_model = whisper_model.to(args.device)
    ecog_model = WhisperForConditionalGeneration(
        WhisperConfig(
            num_mel_bins=args.num_mel_bins,
            max_source_positions=args.max_source_positions,
            d_model=384,
        )
    ).to(args.device)

    # reset decoder and freeze
    ecog_model.model.decoder = whisper_model.model.decoder
    if args.freeze_decoder:
        print("Freezing Decoder")
        for param in ecog_model.model.decoder.parameters():
            param.requires_grad = False

    # change conv stride
    if args.max_source_positions == args.max_neural_len:
        ecog_model.model.encoder.conv2.stride = (1,)

    # init encoder
    for name, param in ecog_model.model.encoder.named_parameters():
        if "weight" in name and param.data.dim() == 2:
            nn.init.kaiming_uniform_(param)

    nn.init.kaiming_normal_(ecog_model.model.encoder.conv1.weight.data, mode="fan_out")
    nn.init.kaiming_normal_(ecog_model.model.encoder.conv2.weight.data, mode="fan_out")

    return ecog_model, processor, tokenizer


def main():
    args = parse_arguments()

    # model1, processor, tokenizer = load_whisper_model_by_hf(args.model_size)
    # model = EcogModel(
    #     EcogConfig(
    #         num_mel_bins=args.num_mel_bins,
    #         max_source_positions=args.max_source_positions,
    #     )
    # ).to(args.device)
    model1, _, _ = load_whisper_model_by_hf(args.model_size)

    model2, processor, tokenizer = load_ecog_model_by_whisper(args)
    breakpoint()

    print("Saving model")
    model2.save_pretrained(args.saving_dir)
    print("Saving processor")
    processor.save_pretrained(args.saving_dir)
    print("Saving tokenizer")
    tokenizer.save_pretrained(args.saving_dir)

    return


if __name__ == "__main__":
    main()
