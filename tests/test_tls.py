"""Which CA bundle the frozen build trusts.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import certifi
import pytest

from paradee_addon import tls


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.delenv('SSL_CERT_DIR', raising=False)


def test_the_system_bundle_comes_first(monkeypatch, tmp_path):
    bundle = tmp_path / 'ca.crt'
    bundle.write_text('x')
    monkeypatch.setattr(tls, 'SYSTEM_CA_FILES', (str(tmp_path / 'missing'), str(bundle)))
    assert tls.configure_ca_bundle(frozen=True, platform='linux') == str(bundle)


def test_no_system_bundle_falls_back_to_certifi(monkeypatch, tmp_path):
    monkeypatch.setattr(tls, 'SYSTEM_CA_FILES', (str(tmp_path / 'missing'),))
    assert tls.configure_ca_bundle(frozen=True, platform='linux') == certifi.where()


def test_a_set_variable_wins_and_source_runs_are_left_alone(monkeypatch):
    assert tls.configure_ca_bundle(frozen=False, platform='linux') is None
    assert tls.configure_ca_bundle(frozen=True, platform='win32') is None
    monkeypatch.setenv('SSL_CERT_DIR', '/somewhere')
    assert tls.configure_ca_bundle(frozen=True, platform='linux') is None
