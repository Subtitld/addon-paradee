# Paradee add-on for Subtitld

Offline **lightweight English text-to-speech** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by
[Paradee](https://github.com/sahilmahendrakar/paradee) by Sahil Mahendrakar:
an 8M-parameter model distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
run on [ONNX Runtime](https://onnxruntime.ai).

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Natural for its size.** Four of Kokoro's voices, at 24 kHz, close to
  Kokoro itself on naturalness and intelligibility with a tenth of its
  parameters ([arXiv:2610.06817](https://arxiv.org/abs/2610.06817)).
- **Small and fast, CPU only.** One 8.5 MB model for all four voices. On one
  thread, a line comes out about 17 to 20 times faster than real time, text
  analysis included. About 300 MB of memory. No GPU and no torch.
- **Offline after the first line.** The model downloads the first time a line
  is spoken, then nothing leaves the machine: no telemetry, no account.
- **English only,** in American and British accents.

## Voices

| Voice | Accent | Kokoro voice | Notes |
|-------|--------|--------------|-------|
| Heart (American English woman) | American | `af_heart` | The default; closest to Kokoro |
| Puck (American English man) | American | `am_puck` | |
| Isabella (British English woman) | British | `bf_isabella` | The furthest from Kokoro of the four |
| Fable (British English man) | British | `bm_fable` | |

The British voices read British pronunciations, the American ones American,
as Kokoro does. Paradee does not clone voices.

## Options

Set these in **Settings ▸ Add-ons ▸ Paradee ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| CPU threads | automatic | `PARADEE_THREADS` | 0 uses one thread, which is upstream's choice and already fast. |
| Speaking pace | 1.0 | `PARADEE_SPEED` | Paradee's own pace, 0.5 to 2.0: the model speaks faster or slower rather than the audio being stretched. |

Subtitld stretches a dub to fit its subtitle itself, so the add-on ignores the
rate slider; the speaking pace is for changing how Paradee itself talks.

## How it runs

Upstream publishes one int8 ONNX graph that goes from Kokoro's phoneme ids to
audio, and a small Python wrapper. The add-on ports the wrapper:

- `paradee_addon/g2p.py`: the text becomes phonemes exactly as upstream (and
  Kokoro) makes them, with [misaki](https://github.com/hexgrad/misaki),
  Kokoro's own G2P, on [spaCy](https://spacy.io)'s small English pipeline,
  which tags each word so numbers, money, abbreviations and words such as
  "read" or "record" come out right. Paradee only learned misaki's phonemes.
- `paradee_addon/espeak.py`: libespeak-ng through ctypes, for the words in no
  misaki dictionary (names, mostly), set up as misaki's own fallback sets up
  the `phonemizer` package, without that package's dependencies.
- `paradee_addon/engine.py`: the ONNX graph, and upstream's sentence splitting.

`tests/test_g2p.py` checks the phonemes against upstream's on test sentences
in both accents, and misaki's fallback word by word (all identical).
`tests/test_engine.py` checks each voice's audio against upstream's wrapper
for the same text (within one 16-bit step). `tools/make_fixtures.py` makes
both fixtures with upstream's own code.

The model mixes random noise into the voice (its breath and hiss). The add-on
seeds that noise, so the same run of lines sounds the same in every session;
a line spoken twice in one session differs only in that noise.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on (bundled) | | GPL-3.0-or-later | this repository |
| espeak-ng 1.52.0, English data only (bundled) | 1.4 MB | GPL-3.0-or-later | [espeak-ng](https://github.com/espeak-ng/espeak-ng) |
| misaki 0.9.4, its English dictionaries (bundled) | 13 MB | Apache-2.0 | [hexgrad/misaki](https://github.com/hexgrad/misaki) |
| spaCy 3.8 and `en_core_web_sm` 3.8.0 (bundled) | 65 MB | MIT | [explosion/spaCy](https://github.com/explosion/spaCy) |
| ONNX Runtime, numpy, num2words and what spaCy imports (bundled) | see the release | MIT, BSD, Apache-2.0; num2words LGPL-2.1; certifi and tqdm MPL-2.0 | PyPI |
| Paradee 1.2, int8 (downloaded) | 8.5 MB | Apache-2.0 | [sahilmahendrakar/Paradee-8M-v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) |

**Why GPL-3.0-or-later.** The add-on says unknown words with espeak-ng, which
is GPL-3.0-or-later, and ships it in the same program; so the add-on is
GPL-3.0-or-later too. Paradee, its weights and misaki remain Apache-2.0.
The complete corresponding source of each release, with espeak-ng's,
num2words', certifi's and tqdm's, is `paradee-<version>-source.tar.gz` on the
same GitHub release.

On Linux the build compiles espeak-ng itself from the 1.52.0 commit
(`tools/build_espeak.sh`), because the espeakng-loader wheel's library needs
glibc 2.29 and the add-on runs on 2.28. macOS and Windows use the wheel's
library, built from the same commit.

The model goes to Subtitld's model folder
(`$SUBTITLD_MODELS_DIR/paradee/<model>/<revision>`), at the revision in
`paradee_addon/pins.json`, each file checked by its sha256. An interrupted
download resumes. `$HF_ENDPOINT` points it at a mirror. The frozen Linux and
macOS builds trust the system's CA bundle (`SSL_CERT_FILE` overrides it).

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it; the build fails if one has none. The
Linux archive carries libquadmath (LGPL-2.1), from numpy's wheel; its source
is attached to the same release.

## Limits

- **English only.** A project in another language is refused with a message
  that says so. Any English (`en-AU`, `en-IN`...) is accepted; each voice
  reads it in its own accent.
- **Hard words.** Names, places and words from other languages that misaki
  does not know are said by espeak-ng, and may be mispronounced.
- **Long lines** are read one sentence at a time and joined, as upstream does.
  Prosody is not planned across sentences.
- **Platforms.** Releases ship Linux x86_64 (glibc 2.28 or newer), macOS
  arm64 and Windows x86_64.
- **One line at a time.** A cancel stops the line between sentences, or stops
  the download, which resumes on the next line.

## Build

Development, in a venv of its own:

```bash
pip install -c constraints.txt -e '.[build,dev]'
pytest
```

The Linux release archive, in the AlmaLinux 8 container (needs Docker; commit
first, the Corresponding Source archives git HEAD):

```bash
tools/build_linux.sh
```

It compiles espeak-ng, freezes the bundle with `pyinstaller.spec`, gathers
`THIRD_PARTY_LICENSES/`, and writes `paradee-<version>-linux-x86_64.zip`,
the libquadmath source and the Corresponding Source tarballs. Then:

```bash
PARADEE_TEST_BINARY=$PWD/dist/paradee-addon/paradee-addon pytest tests/test_installed.py
```

On macOS or Windows, stage espeak-ng from the wheel, then freeze:

```bash
python tools/stage_espeak.py --loader
pyinstaller pyinstaller.spec --noconfirm
dist/paradee-addon/paradee-addon --self-test
```

To move to another model revision, run `tools/pin_model.py <sha>`, then
regenerate the fixtures with `tools/make_fixtures.py` in a venv with upstream's
package, and check that the tests still pass.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).

## Licence

GPL-3.0-or-later; see [LICENSE](LICENSE).
