
"""Validate CS-Dialogue samples using official metadata."""

import csv
import random
import re
import shutil
import wave
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cs_dialogue"

METADATA = DATA / "index" / "Information_Index.txt"
MANIFEST = DATA / "manifest.csv"
DEBUG_MANIFEST = DATA / "debug_manifest.csv"

DEBUG_COUNT_PER_TAG = 12
SEED = 395

FIELDS = [
    "id",
    "audio_path",
    "text",
    "TAG",
    "speaker",
    "split",
    "duration",
]

# Match WAV ID, official TAG, and transcription.
ENTRY_PATTERN = re.compile(
    r"(?P<wav>[^\s]+\.wav)\s+"
    r"(?P<tag><[A-Za-z][A-Za-z0-9_-]*>)"
    r"\s+(?P<script>.*)$"
)


def load_official_metadata():
    records = {}
    skipped = 0

    with METADATA.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("Speaker ID"):
                continue

            speaker_match = re.match(r"^(\d+)\s+", line)
            entry_match = ENTRY_PATTERN.search(line)

            if not speaker_match or not entry_match:
                skipped += 1
                continue

            speaker_id = speaker_match.group(1)
            wav_id = Path(
                entry_match.group("wav").replace("\\", "/")
            ).stem

            record = {
                "speaker": speaker_id,
                "TAG": entry_match.group("tag"),
                "script": entry_match.group("script").strip(),
            }

            if wav_id in records and records[wav_id] != record:
                raise ValueError(
                    f"Conflicting metadata for {wav_id}"
                )

            records[wav_id] = record

    if not records:
        raise RuntimeError("No official metadata was parsed.")

    print(f"Official metadata entries: {len(records)}")
    print(f"Unparsed metadata lines: {skipped}")

    return records


def verify_audio(audio_path):
    path = Path(audio_path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frames = audio.getnframes()

        if rate != 16000:
            raise ValueError(f"Unexpected sample rate: {path}")

        if channels != 1 or sample_width != 2:
            raise ValueError(f"Unexpected WAV format: {path}")

        return round(frames / rate, 3)


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def select_debug_subset(rows):
    rng = random.Random(SEED)
    selected = []
    selected_ids = set()

    tags = ["<CN>", "<MIX>", "<EN>"]
    tags.extend(
        sorted(set(row["TAG"] for row in rows) - set(tags))
    )

    for tag in tags:
        candidates = [
            row for row in rows
            if row["TAG"] == tag
        ]

        candidates.sort(key=lambda row: row["id"])
        rng.shuffle(candidates)

        for row in candidates[:DEBUG_COUNT_PER_TAG]:
            if len(selected) >= 36:
                break

            selected.append(row)
            selected_ids.add(row["id"])

    remaining = [
        row for row in rows
        if row["id"] not in selected_ids
    ]

    remaining.sort(key=lambda row: row["id"])
    rng.shuffle(remaining)

    selected.extend(remaining[:36 - len(selected)])

    return sorted(selected, key=lambda row: row["id"])


def main():
    official = load_official_metadata()
    existing = read_csv(MANIFEST)

    print(f"Existing manifest rows: {len(existing)}")

    updated = []
    missing = []
    tag_mismatches = []
    text_mismatches = []

    tag_mapping = {
        "CN": "<CN>",
        "EN": "<EN>",
        "MIXED": "<MIX>",
        "MIX": "<MIX>",
    }

    for row in existing:
        utterance_id = row["id"]
        metadata = official.get(utterance_id)

        if metadata is None:
            missing.append(utterance_id)
            continue

        old_tag = row["TAG"]
        new_tag = metadata["TAG"]

        if tag_mapping.get(old_tag, old_tag) != new_tag:
            tag_mismatches.append(
                (utterance_id, old_tag, new_tag)
            )

        if (
            re.sub(r"\s+", "", row["text"])
            != re.sub(r"\s+", "", metadata["script"])
        ):
            text_mismatches.append(utterance_id)

        duration = verify_audio(row["audio_path"])

        updated.append({
            "id": utterance_id,
            "audio_path": row["audio_path"],
            "text": row["text"],
            "TAG": new_tag,
            "speaker": metadata["speaker"],
            "split": row["split"],
            "duration": duration,
        })

    if missing:
        print("Missing metadata IDs:", missing[:10])
        raise RuntimeError(
            f"{len(missing)} samples have no official metadata."
        )

    if len(updated) != len(existing):
        raise RuntimeError("Manifest row count changed.")

    updated.sort(key=lambda row: row["id"])

    backup = DATA / "manifest.before_official.csv"

    if not backup.exists():
        shutil.copy2(MANIFEST, backup)

    debug_rows = select_debug_subset(updated)

    write_csv(MANIFEST, updated)
    write_csv(DEBUG_MANIFEST, debug_rows)

    print("\n=== Official Metadata Validation ===")
    print(f"Validated WAV files: {len(updated)}")

    print("\nOfficial TAG distribution:")
    for tag, count in Counter(
        row["TAG"] for row in updated
    ).most_common():
        print(f"  {tag}: {count}")

    print("\nOfficial speaker count:")
    speakers = Counter(row["speaker"] for row in updated)
    print(f"  Unique speakers: {len(speakers)}")
    for speaker, count in sorted(speakers.items()):
        print(f"  {speaker}: {count}")

    print("\nSplit distribution:")
    for split, count in Counter(
        row["split"] for row in updated
    ).items():
        print(f"  {split}: {count}")

    print(f"\nTAG mismatches: {len(tag_mismatches)}")
    for item in tag_mismatches[:5]:
        print(f"  {item}")

    print(f"Text differences: {len(text_mismatches)}")
    print("Text remains sourced from the split index.")
    for utterance_id in text_mismatches[:5]:
        print(f"  {utterance_id}")

    print("\nDebug subset distribution:")
    for tag, count in Counter(
        row["TAG"] for row in debug_rows
    ).most_common():
        print(f"  {tag}: {count}")

    print("\nOutput:")
    print(MANIFEST)
    print(DEBUG_MANIFEST)
    print("\nValidation complete.")


if __name__ == "__main__":
    main()
