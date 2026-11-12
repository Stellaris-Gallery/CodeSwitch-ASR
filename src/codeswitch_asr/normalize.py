"""Text normalization, shared by MER scoring, the CTC vocabulary, and switch-point analysis.

The SAME function runs on references (CS-Dialogue / our transcripts) and on model output,
so that two transcripts that say the same thing in different writing styles score 0 errors.

Rules (decided 2026-11-11):
  1. Remove annotation tags: <FIL/> <SPK/> <NON/> <NPS/> (and any other <...> tag).
  2. Remove the unintelligible marker "**". (Training drops such utterances entirely;
     see has_unintelligible(). Test keeps the utterance and only removes the marker.)
  3. Traditional -> Simplified Chinese (Whisper sometimes outputs Traditional).
  4. English -> UPPERCASE (CS-Dialogue writes English in uppercase).
  5. Remove punctuation (Chinese and English). Keep apostrophes inside words: DON'T.
  6. Tokenize: one token per Chinese character, one token per English word.
  7. Merge runs of single letters into one word: "I B M" -> "IBM", so the reference
     style ("I B M") and Whisper's style ("IBM") match.
Digits are left untouched on purpose (see the project notes: digits are handled when
Whisper generates text, not here).
"""

import re
import unicodedata

from opencc import OpenCC

# ---- patterns -----------------------------------------------------------------------

# Any tag written like <FIL/>, <SPK/>, <S>, </S>.
_TAG = re.compile(r"</?[A-Za-z]+/?>")

# Unintelligible marker: two or more asterisks.
_UNINTELLIGIBLE = re.compile(r"\*{2,}")

# Chinese characters: CJK Unified Ideographs, Extension A, and 〇 (used in 二〇二五).
_CJK = "\u3400-\u4dbf\u4e00-\u9fff\u3007"

# Latin letters, incl. accented capitals like É (so CAFÉ stays one word).
_LATIN = "A-Z\u00c0-\u00d6\u00d8-\u00de"

# One token = a single Chinese character, OR a run of letters/digits/apostrophes.
_TOKEN = re.compile(rf"[{_CJK}]|[{_LATIN}0-9']+")

# Curly apostrophes -> straight apostrophe, so DON’T and DON'T are the same word.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})

# Traditional -> Simplified converter (built once; building it is slow).
_T2S = OpenCC("t2s")


# ---- public functions -----------------------------------------------------------------

def has_unintelligible(text: str) -> bool:
    """True if the transcript contains the "**" marker (used to drop training utterances)."""
    return bool(_UNINTELLIGIBLE.search(text))


def tokenize(text: str) -> list[str]:
    """Normalize `text` and split it into MER tokens (Chinese characters + English words)."""
    s = unicodedata.normalize("NFKC", text)    # full-width -> half-width: "ＡＢＣ，" -> "ABC,"
    s = _TAG.sub(" ", s)                       # rule 1: drop annotation tags
    s = _UNINTELLIGIBLE.sub(" ", s)            # rule 2: drop "**"
    s = _T2S.convert(s)                        # rule 3: 這個 -> 这个
    s = s.upper()                              # rule 4: project -> PROJECT
    s = s.translate(_APOSTROPHES)              # DON’T -> DON'T

    # rules 5 + 6: anything that is not a Chinese character / letter / digit / apostrophe
    # (punctuation, spaces, hyphens, emoji ...) simply never becomes part of a token.
    tokens = _TOKEN.findall(s)

    # rule 5 (apostrophes): keep them only *inside* a word: "'COOL'" -> "COOL", "'" -> gone
    tokens = [t.strip("'") for t in tokens]
    tokens = [t for t in tokens if t]

    return _merge_single_letters(tokens)       # rule 7: I B M -> IBM


def normalize(text: str) -> str:
    """Normalized text as one string, tokens joined by single spaces.

    This is the form jiwer scores: it counts edits between space-separated tokens,
    so with Chinese characters spaced out, jiwer's "WER" is exactly our MER.
    """
    return " ".join(tokenize(text))


# ---- helpers --------------------------------------------------------------------------

def _is_single_letter(token: str) -> bool:
    return len(token) == 1 and "A" <= token <= "Z"


def _merge_single_letters(tokens: list[str]) -> list[str]:
    """Join 2+ consecutive single-letter tokens: [我, I, B, M, 的] -> [我, IBM, 的]."""
    out: list[str] = []
    run: list[str] = []                        # current run of single letters
    for t in tokens + [""]:                    # "" = sentinel that flushes the last run
        if _is_single_letter(t):
            run.append(t)
            continue
        if run:
            out.append("".join(run))           # a run of 1 letter stays as it is ("A")
            run = []
        if t:
            out.append(t)
    return out
