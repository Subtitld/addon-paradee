"""manifest.json, pins.json, pyproject and the code agree with each other.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from paradee_addon import ADDON_ID, __version__, voices
from paradee_addon import __main__ as addon

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'paradee_addon' / 'pins.json').read_text(encoding='utf-8'))


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version'] == addon.ADDON_VERSION
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID == 'paradee'


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['compute'] == 'cpu'
    assert MANIFEST['executable'] == 'bin/paradee-addon'
    assert set(MANIFEST['platforms']) == {'linux', 'macos', 'windows'}
    assert MANIFEST['license'].startswith('GPL-3.0-or-later')
    keys = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(keys) == {'threads', 'speed'}
    defaults = addon._options({})
    assert {k: f['default'] for k, f in keys.items()} == defaults


def test_manifest_model_is_the_pinned_one():
    [entry] = MANIFEST['models']
    pinned = PINS['models'][addon.MODEL]
    assert entry['id'] == addon.MODEL and entry.get('default') is True
    assert entry['repo'] == pinned['repo'] and entry['revision'] == pinned['revision']
    assert entry['size_mb'] == round(sum(f['size'] for f in pinned['files']) / 1e6)
    assert entry['license'] == 'Apache-2.0' and entry['bundled'] is False


def test_pins():
    pins = addon.pins()
    assert [p['path'] for p in pins] == ['config.json', 'onnx/paradee_int8.onnx']
    for pin in pins:
        assert re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) and pin['size'] > 1_000, pin
        assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']) and pin['model'] == addon.MODEL, pin
    assert addon.model_dir() == addon.download.local_path(pins[1]).parent.parent


def test_self_test_text_reaches_the_fallback():
    """The bundle's self-test needs espeak-ng: one word of its text is in no misaki dictionary."""
    assert 'Zxqwerty' in addon.SELF_TEST_TEXT
