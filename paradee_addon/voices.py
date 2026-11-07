"""The add-on's voices: the four Kokoro voices Paradee 1.2 learned.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

One 8.5 MB model speaks all four; the voice picks a row of its voice table.
The two British voices read British phonemes (misaki's British dictionary),
as Kokoro does, and the American ones American phonemes.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 24000
LANGUAGES = ('en-us', 'en-gb')


@dataclass(frozen=True)
class Voice:
    id: str
    kokoro: str          # Kokoro's name for it, which the model's config.json lists
    language: str
    display_name: str

    @property
    def british(self) -> bool:
        return self.language == 'en-gb'


VOICES = (
    Voice('heart', 'af_heart', 'en-us', 'Heart (American English woman)'),
    Voice('puck', 'am_puck', 'en-us', 'Puck (American English man)'),
    Voice('isabella', 'bf_isabella', 'en-gb', 'Isabella (British English woman)'),
    Voice('fable', 'bm_fable', 'en-gb', 'Fable (British English man)'),
)
DEFAULT = VOICES[0]


def by_id(voice_id: str) -> Voice | None:
    """The voice, by its id or its Kokoro name. '' gives the default."""
    if not voice_id:
        return DEFAULT
    return next((v for v in VOICES if voice_id in (v.id, v.kokoro)), None)


def primary_language(raw) -> str:
    """'en' from 'en-US', 'en_GB', 'EN'..., '' when blank."""
    return str(raw or '').strip().replace('_', '-').split('-', 1)[0].lower()


def manifest_voices() -> list[dict]:
    return [{'id': v.id, 'language': v.language, 'display_name': v.display_name, 'sample_rate': SAMPLE_RATE}
            for v in VOICES]
