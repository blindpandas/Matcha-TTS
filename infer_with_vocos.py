# coding: utf-8

import argparse
import datetime as dt
import os
import time
import warnings
from pathlib import Path

import soundfile as sf
import torch

from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse


TEXT = """
Every tribe has its myths, and the younger members of the tribe generally get them wrong. My tribal myth of the great Berkeley Fire of 1923 went this way: when my mother's mother-in-law, who lived near the top of Cedar Street, saw the flames sweeping over the hill straight towards the house, she put her Complete Works of Mark Twain in Twenty-Five Volumes into her Model A and went away from that place. Because I was going to put that story in print, I made the mistake of checking it first with my brother Ted. In a slow, mild sort of way, Ted took it all to pieces. He said, well, Lena Brown never had a Model A. As a matter of fact, she didn't drive. The way I remember the story, he said, some fraternity boys came up the hill and got her piano out just before the fire reached that hill. And a bearskin rug, and some other things. But I don't remember, he said, that anything was said about the Complete Works of Mark Twain.
""".strip()

def process_text(text: str, device: torch.device):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners_piper"]), 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return x, x_lengths


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.eval()
    print(f"[+] {model_name} loaded!")
    return model


def save_to_folder(filename, waveform, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    output_filepath = folder.joinpath(f"{filename}.wav")
    waveform = waveform.squeeze().cpu().numpy()
    sf.write(os.fspath(output_filepath), waveform, 22050,)
    return output_filepath


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser("matcha_infer._vocos.py")

    parser.add_argument("--matcha", type=str, help="Matcha checkpoint", required=True)
    parser.add_argument("--vocos", type=str, help="Vocos .pt", required=True)
    parser.add_argument("--text", type=str, default=TEXT)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temp", type=float, help="Temperature", default=0.667)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--n-timesteps", type=float, default=3)

    args = parser.parse_args()

    device = args.device

    matcha = load_matcha("hfc-female", args.matcha, device)
    vocos = torch.jit.load(args.vocos)

    text = args.text.strip()
    x, x_lengths = process_text(text, device)
    t0 = time.perf_counter()
    output = matcha.synthesise(
        x,
        x_lengths,
        n_timesteps=args.n_timesteps,
        temperature=args.temp,
        length_scale=args.length_scale
    )
    matcha_infer = (time.perf_counter() - t0) * 1000
    mels = output["mel"]
    # Vocode
    t0 = time.perf_counter()
    waveform = vocos(mels).squeeze()
    vocos_infer = (time.perf_counter() - t0) * 1000
    location = save_to_folder("out", waveform, args.out_dir)
    duration = (waveform.shape[0] / 22050) * 1000
    print(f"Audio duration: {duration}")
    print(f"Matcha infer: {matcha_infer}")
    print(f"Vocos infer: {vocos_infer}")
    print(f"Matcha RTF: {matcha_infer / duration}")
    print(f"Vocos RTF: {vocos_infer / duration}")
    print(f"Total RTF: {(vocos_infer / duration) + (matcha_infer / duration)}")
    print(f"Wrote output to folder: {args.out_dir}")


if __name__ == '__main__':
    main()