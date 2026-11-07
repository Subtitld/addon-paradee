"""espeak.py: the library, its two English voices, and the bundled data subset.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

How misaki's fallback uses it is checked against misaki itself in
tests/test_g2p.py (tests/fixtures/fallback.json).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from paradee_addon import espeak

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
WORDS = list(json.loads((FIXTURES / 'fallback.json').read_text(encoding='utf-8')))
ENGLISH_DATA = ('phontab', 'phonindex', 'phondata', 'intonations', 'en_dict', 'lang/gmw/en-US', 'lang/gmw/en')


def test_espeak_is_the_version_upstream_used(espeak_engine):
    assert espeak_engine.version == '1.52.0'


def test_the_two_accents_differ(espeak_engine):
    assert espeak_engine.phonemize('tomato', 'en-us') != espeak_engine.phonemize('tomato', 'en-gb')


@pytest.mark.parametrize('raw, tie, expected', [
    ('Hello, world!', None, 'həlˈoʊ, wˈɜːld!'),
    ('chime', '^', 't^ʃˈa^ɪm'),
    ('', None, ''),
    ('...', None, '...'),
])
def test_punctuation_and_ties(espeak_engine, raw, tie, expected):
    assert espeak_engine.phonemize(raw, 'en-us', tie=tie) == expected


def test_an_unknown_language_is_an_espeak_error(espeak_engine):
    with pytest.raises(espeak.EspeakError):
        espeak_engine.phonemize('bonjour', 'fr-fr')


def test_language_flags_are_dropped():
    assert espeak.strip_language_flags('ðə (fr)ʃɑːtˈoː(en) ɪz') == ('ðə ʃɑːtˈoː ɪz', True)
    assert espeak.strip_language_flags('nˈoʊ flˈæɡz') == ('nˈoʊ flˈæɡz', False)


def test_the_english_data_alone_phonemizes_the_same(tmp_path, espeak_engine):
    """The frozen build ships only these files (tools/stage_espeak.py)."""
    library, data = espeak.locate()
    subset = tmp_path / 'espeak-ng-data'
    for name in ENGLISH_DATA:
        (subset / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(data) / name, subset / name)
    sample = [(word, language, tie) for word in WORDS for language in espeak.VOICES for tie in (None, '^')]
    # A second library instance would share the first one's global state, so
    # compare in a child process.
    import subprocess
    import sys
    code = ('import json, sys; from paradee_addon import espeak; '
            'e = espeak.Espeak(sys.argv[1], sys.argv[2]); '
            'print(json.dumps([e.phonemize(w, l, tie=t) for w, l, t in json.loads(sys.argv[3])]))')
    out = subprocess.run([sys.executable, '-c', code, str(library), str(subset), json.dumps(sample)],
                         capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])))
    assert json.loads(out.stdout) == [espeak_engine.phonemize(w, l, tie=t) for w, l, t in sample]


def test_a_data_path_too_long_for_espeak_is_copied(tmp_path):
    """espeak-ng truncates a data path over 160 bytes without saying so."""
    library, data = espeak.locate()
    deep = tmp_path / ('d' * 60) / ('e' * 60) / 'espeak-ng-data'
    for name in ENGLISH_DATA:
        (deep / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(data) / name, deep / name)
    import subprocess
    import sys
    code = ('import sys; from paradee_addon import espeak; e = espeak.Espeak(sys.argv[1], sys.argv[2]); '
            'print(e._scratch is not None, e.phonemize("hello world")); e.close()')
    out = subprocess.run([sys.executable, '-c', code, str(library), str(deep)], capture_output=True, text=True,
                         check=True, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])))
    assert out.stdout.split() == ['True', 'həlˈoʊ', 'wˈɜːld']


def test_missing_data_is_an_espeak_error(tmp_path):
    library, _data = espeak.locate()
    with pytest.raises(espeak.EspeakError):
        espeak.Espeak(library, tmp_path)
