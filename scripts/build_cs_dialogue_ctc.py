"""Build a train-only character-level CTC vocabulary and training manifest.

Run from the CodeSwitch-ASR repository:
    uv run python scripts/build_cs_dialogue_ctc.py --dataset-root "D:\\Deeplearning Final Project\\Datasets\\CS-Dialogue"

The official CS-Dialogue train split is the ONLY source used to build the
vocabulary. Transcripts containing ** are excluded from training and counted.
The same codeswitch_asr.normalize.tokenize() used for MER is used first;
MER word tokens are then expanded into individual CTC characters.

Writes to <dataset-root>/processed/ only. Does not modify original data,
project dependencies, or Git-tracked files beyond this script itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import wave
from collections import Counter
from pathlib import Path

from codeswitch_asr.normalize import has_unintelligible, tokenize

EXPECTED_TRAIN_ROWS = 26_239
ALLOWED_TAGS = {"<CN>", "<EN>", "<MIX>"}
# The dataset's official metadata file uses this spelling (with "information").
META_PATTERN = re.compile(
    r"(?P<wav>[^\s]+\.wav)\s+"
    r"(?P<tag><[A-Za-z][A-Za-z0-9_-]*>)\s+(?P<script>.*)$"
)

TRAIN_FIELDS = [
    "id", "audio_path", "text", "TAG", "speaker", "split", "duration",
    "normalized_text", "ctc_text",
]
DROP_FIELDS = ["id", "text", "TAG", "speaker", "reason"]


def read_kaldi(path: Path) -> dict[str, str]:
    """Parse a Kaldi key/value file, rejecting malformed and duplicate IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Required official index is missing: {path}")
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed Kaldi line {number}: {path}")
            key, value = parts
            if key in result:
                raise ValueError(f"Duplicate index ID {key}: {path}")
            result[key] = value
    return result


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Get the OFFICIAL speaker and TAG, not an inferred language label."""
    if not path.is_file():
        raise FileNotFoundError(f"Required official speaker/TAG metadata is missing: {path}")
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("Speaker ID"):
                continue
            speaker = re.match(r"^(\d+)\s+", stripped)
            entry = META_PATTERN.search(stripped)
            if not speaker or not entry:
                continue
            key = Path(entry.group("wav").replace("\\", "/")).stem
            item = {"speaker": speaker.group(1), "TAG": entry.group("tag")}
            if key in metadata:
                if metadata[key] != item:
                    raise ValueError(f"Conflicting metadata for {key}")
                raise ValueError(f"Duplicate metadata entry for {key}")
            metadata[key] = item
    if not metadata:
        raise ValueError("Official metadata parsed no entries")
    return metadata


def index_audio(folder: Path) -> tuple[dict[str, Path], set[str]]:
    """Index extracted short WAVs by filename without loading audio into memory."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Extracted short-wave audio directory is missing: {folder}")
    indexed: dict[str, Path] = {}
    duplicate_ids: set[str] = set()
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            if path.stem in indexed:
                duplicate_ids.add(path.stem)
            else:
                indexed[path.stem] = path
    return indexed, duplicate_ids


def audio_duration(path: Path) -> float:
    """Validate the project's expected mono 16-kHz 16-bit PCM WAV format."""
    with wave.open(str(path), "rb") as audio:
        actual = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        if actual != (16_000, 1, 2):
            raise ValueError(f"Unsupported WAV {path.name}: rate/channels/sample_width={actual}")
        return round(audio.getnframes() / audio.getframerate(), 3)


def ctc_characters(tokens: list[str]) -> str:
    """Turn normalized MER tokens into character-level CTC training text.

    Example: ['我', '用', 'IBM'] -> '我|用|IBM'. Each letter in IBM is
    its own CTC class; 'IBM' is NOT a class in the vocabulary.
    """
    return "|".join(tokens)


def make_vocab(ctc_texts: list[str]) -> dict[str, int]:
    """Create deterministic Wav2Vec2 character-to-ID vocabulary.

    `|` represents a space between MER tokens. `[UNK]` covers unseen
    dev/test characters. `<blank>` is the CTC blank and MUST also be
    configured as the Hugging Face tokenizer's pad token.
    """
    alphabet = set("".join(ctc_texts))
    vocab: dict[str, int] = {"|": 0}
    for char in sorted(alphabet - {"|"}):
        vocab[char] = len(vocab)
    for special in ("[UNK]", "<blank>"):
        if special in vocab:
            raise ValueError(f"Special token collision: {special}")
        vocab[special] = len(vocab)
    return vocab


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(1024 * 1024):
            sha.update(data)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    index_dir = dataset_root / "data" / "index" / "short_wav"
    train_text_file = index_dir / "train" / "text"
    texts = read_kaldi(train_text_file)
    scp = read_kaldi(index_dir / "train" / "wav.scp")
    if set(texts) != set(scp):
        raise ValueError(
            "Train text/wav.scp IDs do not match: "
            f"text-only={len(set(texts) - set(scp))}, "
            f"scp-only={len(set(scp) - set(texts))}"
        )
    if len(texts) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS:,} train entries, got {len(texts):,}")
    # Guard against accidental train/dev/test mixing.
    for split in ("dev", "test"):
        other = read_kaldi(index_dir / split / "text")
        overlap = set(texts).intersection(other)
        if overlap:
            raise ValueError(f"Train/{split} ID leakage: {sorted(overlap)[:5]}")

    metadata = load_metadata(
        dataset_root / "data" / "index" / "total_information" / "Information_Index.txt"
    )
    missing_meta = set(texts) - set(metadata)
    if missing_meta:
        raise ValueError(f"Missing official metadata: {sorted(missing_meta)[:5]}")

    indexed_audio, duplicate_audio = index_audio(dataset_root / "audio" / "short_wav")
    ambiguous = set(texts) & duplicate_audio
    missing_audio = set(texts) - set(indexed_audio)
    if ambiguous or missing_audio:
        raise ValueError(
            f"Ambiguous train audio IDs: {sorted(ambiguous)[:5]}; "
            f"missing train WAVs: {sorted(missing_audio)[:5]}"
        )

    retained: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    targets: list[str] = []
    tag_counts: Counter[str] = Counter()
    excluded_by_tag: Counter[str] = Counter()
    empty_by_tag: Counter[str] = Counter()
    char_counts: Counter[str] = Counter()

    for utterance_id in sorted(texts):
        original = texts[utterance_id]
        info = metadata[utterance_id]
        tag = info["TAG"]
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"Unknown official TAG {tag}: {utterance_id}")
        tag_counts[tag] += 1

        if has_unintelligible(original):
            excluded_by_tag[tag] += 1
            dropped.append({
                "id": utterance_id, "text": original, "TAG": tag,
                "speaker": info["speaker"], "reason": "has_unintelligible",
            })
            continue

        tokens = tokenize(original)  # Shared normalization: single source of truth.
        if not tokens:
            empty_by_tag[tag] += 1
            dropped.append({
                "id": utterance_id, "text": original, "TAG": tag,
                "speaker": info["speaker"], "reason": "empty_after_normalization",
            })
            continue

        normalized_text = " ".join(tokens)
        target = ctc_characters(tokens)
        path = indexed_audio[utterance_id]
        relative_audio = path.relative_to(dataset_root).as_posix()
        retained.append({
            "id": utterance_id,
            "audio_path": relative_audio,
            "text": original,
            "TAG": tag,
            "speaker": info["speaker"],
            "split": "train",
            "duration": audio_duration(path),
            "normalized_text": normalized_text,
            "ctc_text": target,
        })
        targets.append(target)
        char_counts.update(target)

    vocab = make_vocab(targets)
    if len(retained) + len(dropped) != EXPECTED_TRAIN_ROWS:
        raise AssertionError("Train accounting mismatch")
    if len({row["id"] for row in retained}) != len(retained):
        raise AssertionError("Duplicate retained IDs")
    if not retained:
        raise ValueError("No usable training utterances remain")

    stats = {
        "source": "BAAI/CS-Dialogue",
        "split": "train",
        "official_train_rows": EXPECTED_TRAIN_ROWS,
        "train_source_text_sha256": hash_file(train_text_file),
        "kept_train_rows": len(retained),
        "dropped_has_unintelligible": sum(excluded_by_tag.values()),
        "dropped_empty_after_normalization": sum(empty_by_tag.values()),
        "dropped_total": len(dropped),
        "raw_tag_counts": dict(sorted(tag_counts.items())),
        "dropped_unintelligible_by_tag": dict(sorted(excluded_by_tag.items())),
        "dropped_empty_by_tag": dict(sorted(empty_by_tag.items())),
        "ctc_vocab_size": len(vocab),
        "ctc_character_count_including_delimiters": sum(char_counts.values()),
        "ctc_character_count_by_symbol": dict(sorted(char_counts.items())),
        "vocab_source": "train only, after dropping ** and empty normalized transcripts",
        "normalization": "codeswitch_asr.normalize.tokenize",
        "ctc_labeling": "each Chinese character / English letter / digit / apostrophe is one class; | separates MER tokens",
        "word_delimiter_token": "|",
        "unk_token": "[UNK]",
        "ctc_blank_and_pad_token": "<blank>",
        "note": "For Wav2Vec2CTCTokenizer set pad_token='<blank>' and word_delimiter_token='|'.",
    }

    output_dir = dataset_root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "train_ctc_manifest.csv"
    dropped_path = output_dir / "train_excluded.csv"
    vocab_path = output_dir / "vocab.json"
    stats_path = output_dir / "train_ctc_stats.json"
    write_csv(manifest_path, TRAIN_FIELDS, retained)
    write_csv(dropped_path, DROP_FIELDS, dropped)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== TRAIN-ONLY CTC PREPARATION COMPLETE ===")
    print(f"Official train rows: {len(texts):,}")
    print(f"Excluded (**): {sum(excluded_by_tag.values()):,}")
    print(f"Excluded (empty after normalization): {sum(empty_by_tag.values()):,}")
    print(f"Retained train rows: {len(retained):,}")
    print(f"Vocabulary size (incl. |, [UNK], <blank>): {len(vocab):,}")
    print(f"Official train TAG counts: {dict(sorted(tag_counts.items()))}")
    print(f"** excluded by TAG: {dict(sorted(excluded_by_tag.items()))}")
    print(f"Output manifest: {manifest_path}")
    print(f"Output vocabulary: {vocab_path}")
    print(f"Output audit: {dropped_path}")
    print(f"Output statistics: {stats_path}")
    print("No source data or existing project files were modified.")


if __name__ == "__main__":
    main()
