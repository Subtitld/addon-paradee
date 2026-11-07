"""Write tests/fixtures/ from upstream Paradee and misaki themselves.

Copyright (C) 2026 Subtitld contributors
SPDX-License-Identifier: GPL-3.0-or-later

Run it in a venv of its own with upstream's package installed
(`pip install git+https://github.com/sahilmahendrakar/paradee@v1.2.1`, which
brings misaki, spaCy, phonemizer-fork and espeakng-loader), never the add-on's,
and point it at an espeak-ng 1.52.0 build (the add-on bundles that version;
tools/build_espeak.sh makes one):

    python tools/make_fixtures.py --espeak-prefix build/espeak-ng/prefix

Writes:
  phonemes.json         text -> upstream's phonemes (misaki + spaCy + misaki's
                        espeak fallback), American and British, for SENTENCES
  fallback.json         word -> misaki's EspeakFallback output, both accents,
                        for FALLBACK_WORDS (words in no misaki dictionary)
  parity-<voice>.wav    upstream's Paradee(...)(PARITY_TEXT, voice=<voice>), the
                        first line of a new Paradee each, made after
                        onnxruntime.set_seed(0) as the add-on does: the model
                        mixes in random noise, drawn from that seed
  parity.json           what produced them

The sentences are the add-on's own, written for these tests.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
PARITY_TEXT = "Good evening. It's 7:30, and Dr. Okafor's flight lands at Gate 12!"
PARITY_SPEED = 1.1
SENTENCES = [
    # Subtitle lines.
    "I can't do this anymore.",
    "Wait, where are you going?",
    "We'll be there in five minutes, I promise.",
    "Don't you dare tell her what happened!",
    "Okay... okay. Let's think about this.",
    "Mom? Dad? Is anyone home?",
    "You've got to be kidding me.",
    "Hey! Get back here!",
    "I'd rather not talk about it right now.",
    "She said, \"Leave the door open,\" and walked away.",
    "Thank you — really, thank you.",
    "No. No, no, no. That's not what I meant.",
    "Run!",
    "Mm-hmm. Sure.",
    # Numbers, money, times, dates, ordinals.
    "The meeting starts at 10:45 on March 3rd, 2027.",
    "It costs $19.99, or £15, or about €17.50.",
    "He was born in 1984 and moved out in '02.",
    "Turn to page 312 and read the first 25 lines.",
    "We sold 1,250,000 copies in 3 weeks.",
    "The temperature dropped to -5 degrees.",
    "She finished 2nd, and he came 21st.",
    "Call me at 555-0142 after 6 p.m.",
    "About 50% of the votes were counted by 11:59.",
    "Version 2.0 shipped in Q3.",
    # Abbreviations and titles.
    "Dr. Smith and Mrs. Jones met Mr. Lee at St. Mary's.",
    "The FBI, the CIA and NASA were all there.",
    "Lt. Col. Harris reported to Gen. Ortiz.",
    "Meet me at 42 Baker St., apt. 3B.",
    # Words whose sound depends on their role.
    "I read the book yesterday, and I will read it again tomorrow.",
    "They record a new record every year.",
    "Please close the door; we are close to the end.",
    "The wind will wind the kite string around the tree.",
    "Lead the team, and keep the lead pipe away from the water.",
    "He wound the bandage around the wound.",
    "I live here, and the band plays live tonight.",
    "Does he know how many does are in the forest?",
    "Present the present to the guest of honour.",
    "The bass player caught a bass.",
    # Spellings and punctuation.
    "The colour of the theatre's centre stage was grey.",
    "The color of the theater's center stage was gray.",
    "Well — I mean — it's complicated (very complicated).",
    "Hyphenated words: state-of-the-art, mother-in-law, twenty-one.",
    "What?! You did WHAT?",
    "Look: one, two, three... go!",
    "\"Really?\" \"Yes, really.\"",
    "Rock 'n' roll never dies.",
    # Names and words in no dictionary: espeak-ng says them.
    "Zxqwerty Lindqvist met Nguyen-Okafor at the Pfizer office.",
    "Siobhan and Aoife flew from Reykjavík to Ouagadougou.",
    "Tchaikovsky and Dvořák were on the programme.",
    "Szczepański ordered the bánh mì.",
    "Llanfairpwll is a village in Wales.",
    # Long sentences, cut between words when the phonemes do not fit.
    ("This is a deliberately long sentence that keeps going, clause after clause, without a single full stop, "
     "so that its phonemes run past the five hundred and ten positions the model can read at once, and the "
     "add-on has to cut it between two words exactly where upstream cuts it, at the last space that still "
     "fits, and then carry on reading the rest of it as if nothing had happened at all, which is the whole "
     "point of this test sentence, and since one long clause was not quite long enough the first time, here is "
     "another one about the weather, the trains that never run on time, the bakery on the corner that sells "
     "bread before dawn, and the neighbours who argue about all of it over the fence every single evening."),
]
FALLBACK_WORDS = ('Zxqwerty Lindqvist Nguyen Okafor Pfizer Llanfairpwll Szczepański Xochitl Tchaikovsky Rzeszów '
                  'Hjelmslev Aoife Siobhan Dvořák Ouagadougou Kyiv Reykjavík Qatar Wojciech mmmhmm brrr shh grr '
                  'Gdańsk Ljubljana Nkosazana Thiruvananthapuram').split()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--espeak-prefix', type=Path, required=True)
    args = parser.parse_args(argv)
    prefix = args.espeak_prefix.resolve()
    lib_dir = prefix / 'lib64' if (prefix / 'lib64').is_dir() else prefix / 'lib'

    # misaki.espeak points phonemizer at espeakng-loader's library; point
    # espeakng-loader at the 1.52.0 build first.
    import espeakng_loader
    espeakng_loader.get_library_path = lambda: str(lib_dir / 'libespeak-ng.so.1')
    espeakng_loader.get_data_path = lambda: str(prefix / 'share' / 'espeak-ng-data')

    import numpy as np
    import onnxruntime
    from misaki import espeak as misaki_espeak
    from misaki.token import MToken
    from paradee import SAMPLE_RATE, Paradee

    tts = Paradee(version='1.2', quantized=True, threads=1)
    phonemes = {text: {'us': tts.phonemize(text, 'af_heart'), 'gb': tts.phonemize(text, 'bf_isabella')}
                for text in SENTENCES}
    fallbacks = {}
    for british, key in ((False, 'us'), (True, 'gb')):
        fallback = misaki_espeak.EspeakFallback(british=british)
        for word in FALLBACK_WORDS:
            ps, rating = fallback(MToken(text=word, tag='NNP', whitespace=' '))
            fallbacks.setdefault(word, {})[key] = [ps, rating]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'phonemes.json').write_text(json.dumps(phonemes, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (OUT / 'fallback.json').write_text(json.dumps(fallbacks, ensure_ascii=False, indent=1) + '\n',
                                       encoding='utf-8')
    samples = {}
    for voice in tts.voices:
        onnxruntime.set_seed(0)
        audio = Paradee(version='1.2', quantized=True, threads=1)(PARITY_TEXT, speed=PARITY_SPEED, voice=voice)
        pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        with wave.open(str(OUT / f'parity-{voice}.wav'), 'wb') as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(SAMPLE_RATE)
            clip.writeframes(pcm.tobytes())
        samples[voice] = len(audio)
    from importlib.metadata import version
    parity = {'text': PARITY_TEXT, 'speed': PARITY_SPEED, 'samples': samples,
              'phonemes': {'us': tts.phonemize(PARITY_TEXT, 'af_heart'), 'gb': tts.phonemize(PARITY_TEXT, 'bm_fable')},
              'made_with': {name: version(name) for name in ('paradee', 'misaki', 'spacy', 'en_core_web_sm',
                                                              'onnxruntime', 'phonemizer-fork')}}
    (OUT / 'parity.json').write_text(json.dumps(parity, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'wrote {len(phonemes)} sentences, {len(fallbacks)} fallback words and {len(samples)} parity clips to {OUT}')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
