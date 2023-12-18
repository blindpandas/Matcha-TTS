# coding: utf-8

import os
from pathlib import Path


# Change this path if appropriate
ROOT_PATH = Path("/mnt/f/vocos-train/hfc-female/hi-fi-captain/en-US/female/")
# ----------
WAV_PATH = ROOT_PATH.joinpath("wav")
TEXT_PATH = ROOT_PATH.joinpath("wav")


TRAIN_DATASET = []
for ut_type in ["train_parallel", "train_non_parallel"]:
    text_path = TEXT_PATH.joinpath(f"{ut_type}.txt")
    wav_path = WAV_PATH.joinpath(ut_type)
    for line in text_path.read_text(encoding="utf-8").splitlines():
        file_stem, text = [c.strip() for c in line.split("|")]
        filepath = os.fspath(
            wav_path.joinpath(file_stem).with_suffix(".wav")
        )
        assert os.path.isfile(filepath), f"File {filepath} does not exist"
        TRAIN_DATASET.append(f"{filepath} | {text}")

ROOT_PATH.joinpath("train.txt").write_text(
    "\n".join(TRAIN_DATASET),
    encoding="utf-8",
    newline="\n"
)

EVAL_DATASET = []
eval_wav = WAV_PATH.joinpath("eval")
eval_lines = TEXT_PATH.joinpath("eval.txt").read_text(encoding="utf-8").splitlines()
for line in eval_lines:
    file_stem, text = [c.strip() for c in line.split("|")]
    filepath = os.fspath(
        eval_wav.joinpath(file_stem).with_suffix(".wav")
    )
    assert os.path.isfile(filepath), f"File {filepath} does not exist"
    EVAL_DATASET.append(f"{filepath} | {text}")

ROOT_PATH.joinpath("eval.txt").write_text(
    "\n".join(EVAL_DATASET),
    encoding="utf-8",
    newline="\n"
)
