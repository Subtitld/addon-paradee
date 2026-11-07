"""Paradee 1.2 on ONNX Runtime: text in, 24 kHz mono float32 out.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

A port of upstream's `paradee/tts.py` (Copyright Sahil Mahendrakar,
Apache-2.0). The model is one ONNX graph from Kokoro's phoneme ids to audio:

    input_ids  int64 [1, T]   phoneme ids, with a 0 pad at each end, T <= 512
    speed      float32 [1]    1.0 is the natural pace
    voice      int64 [1]      the voice's row in config.json's `voices`
    waveform   float32 [1, samples], 24 kHz

The text is read one sentence at a time, as upstream does: it is cut after
. ! ? and at line breaks, each sentence is phonemized (g2p.py), a sentence
whose phonemes do not fit is cut at its last space that does, and the pieces'
audio is joined end to end.

The graph mixes random noise into the sound (RandomNormalLike, the breath and
hiss of the voice). ONNX Runtime draws it from a generator each session seeds
from the process-wide seed, which is random unless set. The add-on sets it
(SEED) before it makes the session, so a run of lines sounds the same in every
process, and the first line is sample for sample what upstream's
`Paradee(...)(text, speed, voice)` says after `onnxruntime.set_seed(SEED)`.
A line said twice in one session differs only in that noise.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 24_000
DEFAULT_SPEED = 1.0
SPEED_RANGE = (0.5, 2.0)
MAX_PHONEMES = 510   # 512 positions, two of them the pads
MODEL_FILE = 'onnx/paradee_int8.onnx'
CONFIG_FILE = 'config.json'
SEED = 0

_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+|\n+')


class Cancelled(Exception):
    """The request was cancelled between two pieces."""


def default_threads() -> int:
    """One: upstream's choice. The model already runs many times faster than real time on one thread."""
    return 1


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def fit(phonemes: str, limit: int = MAX_PHONEMES) -> Iterator[str]:
    """`phonemes` in pieces of at most `limit`, cut at the last space that fits."""
    while len(phonemes) > limit:
        cut = phonemes.rfind(' ', 0, limit)
        cut = cut if cut > 0 else limit
        yield phonemes[:cut]
        phonemes = phonemes[cut:].lstrip()
    if phonemes:
        yield phonemes


class Paradee:
    """The model's session and vocabulary, and the phonemizer it reads from."""

    def __init__(self, model_dir: str | os.PathLike, phonemizer, threads: int = 0):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / CONFIG_FILE).read_text(encoding='utf-8'))
        self.vocab: dict[str, int] = config['vocab']
        self.voices: list[str] = list(config['voices'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or default_threads()
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        ort.set_seed(SEED)
        self.session = ort.InferenceSession(str(model_dir / MODEL_FILE), options,
                                            providers=['CPUExecutionProvider'])
        self.phonemizer = phonemizer

    def ids(self, phonemes: str) -> list[int]:
        """Kokoro's token ids, with the pads; characters outside the vocabulary are dropped, as upstream does."""
        return [0] + [self.vocab[c] for c in phonemes if c in self.vocab][:MAX_PHONEMES] + [0]

    def pieces(self, text: str, british: bool) -> list[str]:
        return [piece for sentence in sentences(text) for piece in fit(self.phonemizer(sentence, british))]

    def run(self, phonemes: str, *, voice: str, speed: float = DEFAULT_SPEED) -> np.ndarray:
        feed = {
            'input_ids': np.array([self.ids(phonemes)], dtype=np.int64),
            'speed': np.array([speed], dtype=np.float32),
        }
        if len(self.voices) > 1:
            feed['voice'] = np.array([self.voices.index(voice)], dtype=np.int64)
        return np.asarray(self.session.run(['waveform'], feed)[0], dtype=np.float32).reshape(-1)

    def synthesize(self, text: str, *, voice: str, british: bool, speed: float = DEFAULT_SPEED,
                   progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> np.ndarray:
        """`voice` is a Kokoro name from config.json; `british` picks misaki's British dictionary."""
        if not text.strip():
            raise ValueError('Text must not be empty.')
        if voice not in self.voices:
            raise ValueError(f'the model has no voice {voice!r}; its voices are {", ".join(self.voices)}')
        speed = min(max(float(speed), SPEED_RANGE[0]), SPEED_RANGE[1])
        pieces = self.pieces(text, british)
        audio: list[np.ndarray] = []
        for index, piece in enumerate(pieces):
            if cancelled is not None and cancelled():
                raise Cancelled()
            audio.append(self.run(piece, voice=voice, speed=speed))
            if progress is not None:
                progress((index + 1) / len(pieces))
        if not audio:
            raise ValueError('The text has nothing Paradee can say.')
        return np.clip(np.concatenate(audio), -1.0, 1.0)
