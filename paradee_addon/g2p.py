"""Text to Paradee's phonemes: misaki, Kokoro's own G2P, as upstream sets it up.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

Paradee only learned misaki's spelling of English, so the add-on phonemizes
exactly as upstream's `paradee/tts.py` (and Kokoro's KPipeline) does:

    misaki.en.G2P(trf=False, british=..., unk='',
                  fallback=misaki.espeak.EspeakFallback(british=...))

spaCy's small English pipeline (tok2vec and tagger only) tokenizes and tags
the text; misaki's dictionaries give each word its phonemes, using the tags
for numbers, currencies, punctuation and words such as "read" or "record";
espeak-ng says the words in no dictionary.

Two things differ from upstream, neither in the phonemes:

- spaCy's model is a bundled package loaded from here, never downloaded:
  misaki's own constructor would fetch it with `spacy.cli.download` when it
  is missing.
- The fallback is a port of misaki 0.9.4's EspeakFallback onto espeak.py,
  so the `phonemizer` package, with its long chain of imports, stays out.
  tests/test_g2p.py checks it against misaki's own.
"""

from __future__ import annotations

import re
import threading

from misaki import en

# misaki.espeak.EspeakFallback.E2M, misaki 0.9.4 (Apache-2.0, hexgrad): espeak's
# IPA to misaki's, longest first.
_E2M = sorted({
    'ʔˌn̩': 'ʔn', 'ʔn̩': 'ʔn',
    'a^ɪ': 'I', 'a^ʊ': 'W',
    'd^ʒ': 'ʤ',
    'e^ɪ': 'A', 'e': 'A',
    't^ʃ': 'ʧ',
    'ɔ^ɪ': 'Y',
    'ə^l': 'ᵊl',
    'ʲo': 'jo', 'ʲə': 'jə', 'ʲ': '',
    'ɚ': 'əɹ',
    'r': 'ɹ',
    'x': 'k', 'ç': 'k',
    'ɐ': 'ə',
    'ɬ': 'l',
    '̃': '',
}.items(), key=lambda kv: -len(kv[0]))
_SYLLABIC_RE = re.compile(r'(\S)̩')


class EspeakFallback:
    """misaki.espeak.EspeakFallback (misaki 0.9.4) on the add-on's espeak-ng."""

    def __init__(self, espeak, british: bool, version: str | None = None):
        self.espeak = espeak
        self.british = british
        self.version = version
        self.language = 'en-gb' if british else 'en-us'

    def __call__(self, token):
        ps = self.espeak.phonemize(token.text, self.language, tie='^')
        if not ps:
            return None, None
        ps = ps.strip()
        for old, new in _E2M:
            ps = ps.replace(old, new)
        ps = _SYLLABIC_RE.sub(r'ᵊ\1', ps).replace(chr(809), '')
        if self.british:
            ps = ps.replace('e^ə', 'ɛː')
            ps = ps.replace('iə', 'ɪə')
            ps = ps.replace('ə^ʊ', 'Q')
        else:
            ps = ps.replace('o^ʊ', 'O')
            ps = ps.replace('ɜːɹ', 'ɜɹ')
            ps = ps.replace('ɜː', 'ɜɹ')
            ps = ps.replace('ɪə', 'iə')
            ps = ps.replace('ː', '')
        ps = ps.replace('o', 'ɔ')  # for espeak < 1.52
        if self.version != '2.0':
            ps = ps.replace('ɾ', 'T').replace('ʔ', 't')
        return ps.replace('^', ''), 2


class G2P(en.G2P):
    """misaki's G2P on a spaCy pipeline the caller loaded.

    Sets the same attributes as en.G2P.__init__(version=None, trf=False,
    british=british, fallback=fallback, unk='') in misaki 0.9.4, without its
    download.
    """

    def __init__(self, nlp, british: bool, fallback):
        self.version = None
        self.british = british
        self.nlp = nlp
        self.lexicon = en.Lexicon(british)
        self.fallback = fallback
        self.unk = ''


def load_spacy():
    """spaCy's small English pipeline with the components misaki enables (trf=False)."""
    import en_core_web_sm
    return en_core_web_sm.load(enable=['tok2vec', 'tagger'])


class Phonemizer:
    """One spaCy pipeline, and a G2P per accent, made the first time each is needed."""

    def __init__(self, espeak, nlp=None):
        self.espeak = espeak
        self._nlp = nlp
        self._g2p: dict[bool, G2P] = {}
        self._lock = threading.Lock()

    def g2p(self, british: bool) -> G2P:
        with self._lock:
            if british not in self._g2p:
                if self._nlp is None:
                    self._nlp = load_spacy()
                self._g2p[british] = G2P(self._nlp, british, EspeakFallback(self.espeak, british))
            return self._g2p[british]

    def __call__(self, text: str, british: bool = False) -> str:
        phonemes, _ = self.g2p(british)(text)
        return phonemes
