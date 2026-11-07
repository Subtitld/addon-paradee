"""Real synthesis with the pinned model, from source, against upstream.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

The model (8.5 MB) downloads into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run, checked against pins.json.
tests/fixtures/parity-<voice>.wav is upstream's own `Paradee` speaking the
same text at the same speed, as the first line of a new session seeded as
the add-on seeds it (tools/make_fixtures.py): the model mixes in random noise.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from paradee_addon import __main__ as addon, download, engine, g2p, voices

pytestmark = pytest.mark.model

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
PARITY = json.loads((FIXTURES / 'parity.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def phonemizer(espeak_engine):
    return g2p.Phonemizer(espeak_engine)


@pytest.fixture(scope='module')
def paradee(phonemizer):
    download.ensure(addon.pins(), lambda *_: None, lambda: False)
    return engine.Paradee(addon.model_dir(), phonemizer, threads=1)


def _read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as clip:
        assert clip.getframerate() == engine.SAMPLE_RATE
        return np.frombuffer(clip.readframes(clip.getnframes()), dtype='<i2').astype(np.float32) / 32767.0


def test_the_model_has_the_add_on_s_voices(paradee):
    assert sorted(paradee.voices) == sorted(v.kokoro for v in voices.VOICES)


@pytest.mark.parametrize('voice', voices.VOICES, ids=lambda v: v.id)
def test_matches_upstream_sample_for_sample(paradee, phonemizer, voice):
    accent = 'gb' if voice.british else 'us'
    assert paradee.pieces(PARITY['text'], voice.british) == [
        piece for sentence in engine.sentences(PARITY['text'])
        for piece in engine.fit(paradee.phonemizer(sentence, voice.british))]
    assert ' '.join(paradee.pieces(PARITY['text'], voice.british)) == PARITY['phonemes'][accent]
    fresh = engine.Paradee(addon.model_dir(), phonemizer, threads=1)
    wav = fresh.synthesize(PARITY['text'], voice=voice.kokoro, british=voice.british, speed=PARITY['speed'])
    reference = _read(FIXTURES / f'parity-{voice.kokoro}.wav')
    assert len(wav) == len(reference) == PARITY['samples'][voice.kokoro]
    # The fixture is 16-bit, and ONNX Runtime's int8 kernels round a little
    # differently from one CPU to another: a few 16-bit steps at most (CI
    # runners differ by up to 9).
    assert np.abs(wav - reference).max() < 1e-3
    assert np.corrcoef(wav, reference)[0, 1] > 0.9999


def test_sounds_like_speech(paradee):
    wav = paradee.synthesize('Good morning. Shall we begin the recording?', voice='am_puck', british=False)
    assert wav.dtype == np.float32 and np.abs(wav).max() <= 1.0
    seconds = len(wav) / engine.SAMPLE_RATE
    assert 1.5 < seconds < 5.0
    rms = float(np.sqrt(np.mean(wav ** 2)))
    assert 0.02 < rms < 0.5


def test_a_line_said_twice_differs_only_in_its_noise(paradee):
    line = 'The same line, said twice.'
    a = paradee.synthesize(line, voice='af_heart', british=False)
    b = paradee.synthesize(line, voice='af_heart', british=False)
    assert len(a) == len(b) and not np.array_equal(a, b)
    # The voiced sound, which carries the words, is the same: the two takes
    # are nearly identical overall.
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_speed_changes_the_pace(paradee):
    line = 'Speaking faster makes the line shorter.'
    normal = len(paradee.synthesize(line, voice='bf_isabella', british=True))
    fast = len(paradee.synthesize(line, voice='bf_isabella', british=True, speed=2.0))
    slow = len(paradee.synthesize(line, voice='bf_isabella', british=True, speed=0.5))
    assert fast < 0.65 * normal and slow > 1.6 * normal


def test_a_sentence_too_long_for_the_model_is_cut_between_words(paradee):
    long = max(json.loads((FIXTURES / 'phonemes.json').read_text(encoding='utf-8')), key=len)
    pieces = paradee.pieces(long, british=False)
    assert len(pieces) > 1 and all(len(p) <= engine.MAX_PHONEMES for p in pieces)
    assert ' '.join(pieces) == paradee.phonemizer(long, False)
    wav = paradee.synthesize(long, voice='af_heart', british=False)
    assert len(wav) / engine.SAMPLE_RATE > 20


def test_unknown_voice_and_empty_text_are_refused(paradee):
    with pytest.raises(ValueError):
        paradee.synthesize('Hello.', voice='af_bella', british=False)
    with pytest.raises(ValueError):
        paradee.synthesize('   ', voice='af_heart', british=False)


def test_cancel_between_sentences(paradee):
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(engine.Cancelled):
        paradee.synthesize('First sentence. Second sentence. Third.', voice='af_heart', british=False,
                           cancelled=cancelled)
