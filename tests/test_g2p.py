"""g2p.py against misaki and upstream Paradee themselves.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

tests/fixtures/phonemes.json holds what upstream's phonemizer (misaki 0.9.4,
spaCy's en_core_web_sm, misaki's espeak fallback through phonemizer and
espeak-ng 1.52.0) makes of each test sentence, in both accents;
fallback.json what misaki's EspeakFallback makes of words in no dictionary.
tools/make_fixtures.py wrote both.
"""

from __future__ import annotations

import json
from pathlib import Path

import en_core_web_sm
import pytest
from misaki import en
from misaki.token import MToken

from paradee_addon import g2p

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
PHONEMES = json.loads((FIXTURES / 'phonemes.json').read_text(encoding='utf-8'))
FALLBACK = json.loads((FIXTURES / 'fallback.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def phonemizer(espeak_engine):
    return g2p.Phonemizer(espeak_engine)


@pytest.mark.parametrize('accent', ['us', 'gb'])
def test_sentences_match_upstream(phonemizer, accent):
    assert len(PHONEMES) >= 50
    wrong = {text: (expected[accent], phonemizer(text, british=accent == 'gb'))
             for text, expected in PHONEMES.items()
             if phonemizer(text, british=accent == 'gb') != expected[accent]}
    assert not wrong, f'{len(wrong)} differ, e.g. {next(iter(wrong.items()))}'


@pytest.mark.parametrize('accent', ['us', 'gb'])
def test_the_fallback_matches_misaki_s(espeak_engine, accent):
    fallback = g2p.EspeakFallback(espeak_engine, british=accent == 'gb')
    wrong = {}
    for word, expected in FALLBACK.items():
        got = list(fallback(MToken(text=word, tag='NNP', whitespace=' ')))
        if got != expected[accent]:
            wrong[word] = (expected[accent], got)
    assert not wrong


def test_the_fallback_is_used(phonemizer):
    """Some fixture sentences need it, so the comparison above covers it."""
    _, tokens = phonemizer.g2p(british=False)('Zxqwerty Lindqvist met Nguyen-Okafor.')
    assert any(getattr(token._, 'rating', None) == 2 for token in tokens)


def test_g2p_has_what_misaki_s_constructor_sets(espeak_engine):
    """g2p.G2P skips misaki's __init__ (and its download), so it must set the same attributes."""
    nlp = g2p.load_spacy()
    ours = g2p.G2P(nlp, british=True, fallback=g2p.EspeakFallback(espeak_engine, True))
    theirs = en.G2P(trf=False, british=True, fallback=None, unk='')
    assert set(vars(ours)) == set(vars(theirs))
    assert (ours.version, ours.british, ours.unk) == (theirs.version, theirs.british, theirs.unk)
    assert ours.lexicon.golds == theirs.lexicon.golds


def test_only_the_components_misaki_runs_are_loaded():
    nlp = g2p.load_spacy()
    assert nlp.pipe_names == list(g2p.SPACY_COMPONENTS)
    full = en_core_web_sm.load()
    assert set(full.component_names) == set(g2p.SPACY_COMPONENTS) | set(g2p._SPACY_OTHERS)


def test_empty_text(phonemizer):
    assert phonemizer('', british=False) == ''
