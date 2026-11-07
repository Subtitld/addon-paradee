"""English phonemes from libespeak-ng, called through ctypes.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

Paradee reads misaki's phonemes (g2p.py). For a word that is in none of
misaki's dictionaries, misaki asks the `phonemizer` package's espeak backend,
set up as misaki.espeak.EspeakFallback sets it up:

    EspeakBackend(language='en-us' or 'en-gb', preserve_punctuation=True,
                  with_stress=True, tie='^').phonemize([word])

phonemizer imports its other backends at import time, and with them a long
chain of packages (segments, csvw, clldutils, babel...) the add-on would have
to freeze and license. So this module does the same thing on its own, for that
one configuration (both languages): the text is cut at punctuation marks, each chunk is
phonemized by espeak-ng, the marks are put back, and groups of chunks joined by
quotes are phonemized as a whole as well so their words keep their context
(phonemizer 3.6). tests/test_espeak.py checks the output against phonemizer
itself.

espeak-ng keeps global state, so every call takes one lock.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

log = logging.getLogger('paradee_addon.espeak')

# espeak-ng's names for the two languages misaki's fallback asks for.
VOICES = {'en-us': 'gmw/en-US', 'en-gb': 'gmw/en'}

# espeak_Initialize: AUDIO_OUTPUT_SYNCHRONOUS, no buffer, default options.
_AUDIO_OUTPUT_SYNCHRONOUS = 0x02
# espeak_TextToPhonemes: the text is UTF-8; the phonemes are IPA (bit 1),
# with '_' between the phonemes of a word (bits 8-23), or, in tie mode
# (bit 7), with U+0361 joining the letters of one phoneme.
_CHARS_UTF8 = 1
_PHONEMES_IPA = ord('_') << 8 | 0x02
_PHONEMES_TIE = 0x02 | 0x01 << 7 | ord('\u0361') << 8
_TIE = '\u0361'
# espeak-ng copies its data path into a fixed buffer (N_PATH_HOME, 160 bytes,
# 230 on Windows) and silently truncates a longer one.
_MAX_DATA_PATH = 150

# phonemizer's default marks, and those it treats as inline (quotes and
# brackets, which do not break the utterance).
_MARKS = ';:,.!?¡¿—…"«»“”(){}[]'
_INLINE_MARKS = '"«»“”‘’„‹›()[]{}'
_DECIMAL_SEPARATORS = ',.'
_OTHERS = ''.join(m for m in _MARKS if m not in _DECIMAL_SEPARATORS)
_DECIMALS = re.escape(_DECIMAL_SEPARATORS)
_MARKS_RE = re.compile(
    rf'(\s*(?:[{re.escape(_OTHERS)}]|(?<![0-9])[{_DECIMALS}]|[{_DECIMALS}](?![0-9]))+\s*)+')
# espeak's language switch flags, as in "(fr)".
_FLAGS_RE = re.compile(r'\(.+?\)')
_UNDERSCORES_RE = re.compile(r'_+')


class EspeakError(RuntimeError):
    """libespeak-ng could not be loaded or set up."""


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
class Espeak:
    """libespeak-ng, initialized once; each call picks its English voice."""

    def __init__(self, library: str | os.PathLike, data_dir: str | os.PathLike):
        self._lock = threading.Lock()
        data_dir = Path(data_dir)
        if not (data_dir / 'phontab').is_file():
            raise EspeakError(f'no espeak-ng data at {data_dir}')
        self._scratch = None
        data_path = os.fsencode(data_dir)
        if len(data_path) > _MAX_DATA_PATH:
            # Too long for espeak's buffer: use a copy at a short path.
            self._scratch = tempfile.mkdtemp(prefix='paradee-espeak-')
            shutil.copytree(data_dir, Path(self._scratch) / 'espeak-ng-data')
            data_path = os.fsencode(Path(self._scratch) / 'espeak-ng-data')
            log.info('espeak-ng data path is %d bytes; using a copy in %s', len(os.fsencode(data_dir)),
                     self._scratch)
        try:
            self._lib = ctypes.CDLL(os.fspath(library))
        except OSError as exc:
            raise EspeakError(f'could not load {library}: {exc}') from exc
        lib = self._lib
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        lib.espeak_Info.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        lib.espeak_Info.restype = ctypes.c_char_p
        if lib.espeak_Initialize(_AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path, 0) <= 0:
            raise EspeakError(f'espeak-ng failed to initialize with the data at {data_path!r}')
        self._voice = None
        for language in VOICES:   # both must be there, so a missing one fails at load, not mid-line
            self._set_voice(language)
        self.version = lib.espeak_Info(None).decode('ascii', 'replace').split()[0]

    def _set_voice(self, language: str) -> None:
        """Switch espeak to `language`'s voice. The caller holds the lock, or is __init__."""
        if language == self._voice:
            return
        name = VOICES[language]
        if self._lib.espeak_SetVoiceByName(name.encode('ascii')) != 0:
            raise EspeakError(f'espeak-ng has no voice {name}')
        self._voice = language

    def text_to_phonemes(self, text: str, language: str = 'en-us', tie: bool = False) -> str:
        """espeak's raw IPA: '_' between phonemes (U+0361 inside them with `tie`), ' ' between words."""
        if language not in VOICES:
            raise EspeakError(f'no espeak voice for {language!r}')
        pointer = ctypes.pointer(ctypes.c_char_p(text.encode('utf-8')))
        mode = _PHONEMES_TIE if tie else _PHONEMES_IPA
        result = []
        with self._lock:
            self._set_voice(language)
            while pointer.contents.value is not None:
                phonemes = self._lib.espeak_TextToPhonemes(pointer, _CHARS_UTF8, mode)
                if phonemes:
                    result.append(phonemes.decode('utf-8', 'replace'))
        return ' '.join(result)

    def close(self) -> None:
        if self._scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    # -- phonemizer's espeak backend, one line ------------------------------
    def _phonemize_chunk(self, chunk: str, language: str, tie: str | None) -> str:
        line = self.text_to_phonemes(chunk, language, tie=tie is not None)
        line = line.strip().replace('\n', ' ').replace('  ', ' ')
        if '__' in line:
            line = _UNDERSCORES_RE.sub('_', line)
        line = line.replace('_ ', ' ')
        if not line:
            return ''
        words = [word.strip().replace('_', '') for word in line.split(' ')]
        if tie is not None:
            words = [word.replace(_TIE, tie) for word in words]
        return ' '.join(words)

    def phonemize(self, text: str, language: str = 'en-us', tie: str | None = None) -> str:
        """`text` as phonemizer would phonemize it (see the module docstring).

        `tie` is phonemizer's tie option: the character that joins the letters
        of one phoneme ('^' for misaki), or None for '_'-free plain IPA.
        """
        text = text.strip('\r\n')
        if not text.strip():
            return ''
        chunks, parts = _preserve(text)
        groups = _inline_groups(parts)
        phonemized = [self._phonemize_chunk(chunk, language, tie) for chunk in chunks]
        context = [(group, self._phonemize_chunk(' '.join(chunks[n] for n in group), language, tie))
                   for group in groups]
        return _restore(phonemized, parts, context)


# ---------------------------------------------------------------------------
# phonemizer's Punctuation, for one line, word separator ' ', strip=True
# ---------------------------------------------------------------------------
def _preserve(line: str) -> tuple[list[str], list]:
    """The chunks between the marks, and the line's parts: chunk indices and marks."""
    chunks: list[str] = []
    parts: list = []
    cursor = 0
    for match in _MARKS_RE.finditer(line):
        if line[cursor:match.start()]:
            parts.append(len(chunks))
            chunks.append(line[cursor:match.start()])
        parts.append(match.group())
        cursor = match.end()
    if line[cursor:]:
        parts.append(len(chunks))
        chunks.append(line[cursor:])
    return chunks, parts


def _is_inline(mark: str) -> bool:
    mark = ''.join(mark.split())
    return bool(mark) and all(char in _INLINE_MARKS for char in mark)


def _inline_groups(parts: list) -> list[list[int]]:
    groups = []
    group: list[int] = []
    for num, part in enumerate(parts):
        if not isinstance(part, int):
            continue
        if group and num >= 2 and parts[num - 2] == group[-1] and _is_inline(parts[num - 1]):
            group.append(part)
        else:
            if len(group) > 1:
                groups.append(group)
            group = [part]
    if len(group) > 1:
        groups.append(group)
    return groups


def _words(text: str) -> list[str]:
    return [word for word in text.split(' ') if word]


def _in_context(phonemized: list[str], group: list[int], whole: str) -> list[str] | None:
    """`whole` cut back into the group's chunks, or None when the words do not line up."""
    counts = [len(_words(phonemized[chunk])) for chunk in group]
    if not all(counts) or len(_words(whole)) != sum(counts):
        return None
    starts = set(itertools.accumulate(counts[:-1]))
    pieces = whole.split(' ')
    cuts = [0]
    nwords = 0
    for num, piece in enumerate(pieces):
        if piece:
            if nwords in starts:
                cuts.append(num)
            nwords += 1
    cuts.append(len(pieces))
    return [' '.join(pieces[start:end]) for start, end in zip(cuts, cuts[1:])]


def _restore(phonemized: list[str], parts: list, context: list) -> str:
    position = {part: pos for pos, part in enumerate(parts) if isinstance(part, int)}
    in_context: dict[int, str] = {}
    for group, whole in context:
        places = [position.get(chunk) for chunk in group]
        if None in places or any(second != first + 2 or not _is_inline(parts[first + 1])
                                 for first, second in zip(places, places[1:])):
            continue
        cut = _in_context(phonemized, group, whole)
        if cut is not None:
            in_context.update(zip(group, cut))
    out = ''
    for part in parts:
        if isinstance(part, str):
            if out.endswith(' '):
                out = out[:-1]
            out += part
        else:
            chunk = in_context.get(part, phonemized[part])
            if chunk:
                out += chunk
    return out


def strip_language_flags(phonemes: str) -> tuple[str, bool]:
    """Drop espeak's language switch flags ("(fr)"). True when there were some."""
    if not _FLAGS_RE.search(phonemes):
        return phonemes, False
    return re.sub(r'\s+', ' ', _FLAGS_RE.sub('', phonemes)).strip(), True


# ---------------------------------------------------------------------------
# Where the library and its data are
# ---------------------------------------------------------------------------
def bundled() -> tuple[Path, Path] | None:
    """(library, data dir) shipped beside this module, if any.

    The frozen build carries them in `paradee_addon/espeak/`: on Linux an
    espeak-ng the build compiled (tools/build_espeak.sh), elsewhere the copy
    from the espeakng-loader wheel.
    """
    root = Path(__file__).resolve().parent / 'espeak'
    names = {'win32': 'espeak-ng.dll', 'darwin': 'libespeak-ng.dylib'}
    library = root / names.get(sys.platform, 'libespeak-ng.so.1')
    data = root / 'espeak-ng-data'
    if library.is_file() and data.is_dir():
        return library, data
    return None


def locate() -> tuple[Path, Path]:
    """(library, data dir): $PARADEE_ESPEAK_LIBRARY/_DATA, the bundled copy, or espeakng-loader."""
    library = os.environ.get('PARADEE_ESPEAK_LIBRARY', '').strip()
    data = os.environ.get('PARADEE_ESPEAK_DATA', '').strip()
    if library and data:
        return Path(library), Path(data)
    found = bundled()
    if found:
        return found
    try:
        import espeakng_loader
    except ImportError as exc:
        raise EspeakError('espeak-ng is missing: no bundled copy and no espeakng-loader') from exc
    return Path(espeakng_loader.get_library_path()), Path(espeakng_loader.get_data_path())
