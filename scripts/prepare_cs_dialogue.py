
"""Prepare a small, local CS-Dialogue ASR sample."""

import csv
import io
import os
import random
import re
import tarfile
import wave
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cs_dialogue"
INDEX_DIR = DATA_DIR / "index"
AUDIO_DIR = DATA_DIR / "audio"

ARCHIVE_URL = (
    "https://huggingface.co/datasets/"
    "BAAI/CS-Dialogue/resolve/main/"
    "data/short_wav/short_wav.tar.gz00"
)

TARGET_COUNT = 100
DEBUG_COUNT = 36
MIN_DURATION = 2.0
MAX_DURATION = 15.0
MAX_PER_SPEAKER = 25

FIELDS = [
    "id",
    "audio_path",
    "text",
    "TAG",
    "speaker",
    "split",
    "duration",
]


def read_kaldi_file(path):
    """Read utterance-ID and value pairs."""
    result = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                result[parts[0]] = parts[1]

    return result


def infer_tag(transcript):
    """Estimate language category from transcript text."""
    cleaned = re.sub(r"<[^>]+>", "", transcript)

    has_chinese = bool(
        re.search(r"[\u4e00-\u9fff]", cleaned)
    )
    has_english = bool(
        re.search(r"[A-Za-z]", cleaned)
    )

    if has_chinese and has_english:
        return "MIXED"
    if has_chinese:
        return "CN"
    if has_english:
        return "EN"
    return "UNKNOWN"


def load_index():
    """Combine official split, text and audio indexes."""
    records = {}

    for split in ("train", "dev", "test"):
        folder = INDEX_DIR / split

        texts = read_kaldi_file(folder / "text")
        paths = read_kaldi_file(folder / "wav.scp")

        shared_ids = texts.keys() & paths.keys()

        print(
            f"{split}: "
            f"{len(texts)} transcripts, "
            f"{len(paths)} audio paths, "
            f"{len(shared_ids)} matched"
        )

        for utterance_id in shared_ids:
            records[utterance_id] = {
                "text": texts[utterance_id],
                "source_path": paths[utterance_id],
                "speaker": utterance_id.rsplit("_", 1)[0],
                "split": split,
            }

    return records


def inspect_wav(content):
    """Return WAV duration if format is valid."""
    try:
        with wave.open(io.BytesIO(content), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()

            if rate != 16000:
                return None
            if channels != 1 or sample_width != 2:
                return None

            duration = frames / rate

            if MIN_DURATION <= duration <= MAX_DURATION:
                return round(duration, 3)

    except (wave.Error, EOFError):
        pass

    return None


def write_manifest(path, rows):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_debug_subset(rows):
    """Select reproducible, approximately balanced samples."""
    rng = random.Random(395)
    chosen = []
    used = set()

    for tag in ("MIXED", "CN", "EN"):
        group = [
            row for row in rows
            if row["TAG"] == tag
        ]
        group.sort(key=lambda row: row["id"])
        rng.shuffle(group)

        for row in group[: DEBUG_COUNT // 3]:
            chosen.append(row)
            used.add(row["id"])

    remaining = [
        row for row in rows
        if row["id"] not in used
    ]
    remaining.sort(key=lambda row: row["id"])
    rng.shuffle(remaining)

    chosen.extend(
        remaining[: max(0, DEBUG_COUNT - len(chosen))]
    )

    return sorted(chosen, key=lambda row: row["id"])


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    index = load_index()
    rows = []
    speaker_counts = Counter()

    headers = {"User-Agent": "CS-Dialogue-Research/1.0"}
    token = os.environ.get("HF_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(ARCHIVE_URL, headers=headers)

    print("\nStreaming the first archive segment...")

    try:
        with urlopen(request, timeout=60) as response:
            with tarfile.open(
                fileobj=response, mode="r|gz"
            ) as archive:

                for member in archive:
                    if not member.isfile():
                        continue

                    if not member.name.lower().endswith(".wav"):
                        continue

                    utterance_id = Path(member.name).stem
                    record = index.get(utterance_id)

                    if record is None:
                        continue

                    speaker = record["speaker"]

                    if speaker_counts[speaker] >= MAX_PER_SPEAKER:
                        continue

                    # Reject unusually large files.
                    if member.size > 10 * 1024 * 1024:
                        continue

                    audio_file = archive.extractfile(member)

                    if audio_file is None:
                        continue

                    content = audio_file.read()
                    duration = inspect_wav(content)

                    if duration is None:
                        continue

                    output_path = AUDIO_DIR / f"{utterance_id}.wav"
                    output_path.write_bytes(content)

                    rows.append({
                        "id": utterance_id,
                        "audio_path": output_path.relative_to(
                            ROOT
                        ).as_posix(),
                        "text": record["text"],
                        "TAG": infer_tag(record["text"]),
                        "speaker": speaker,
                        "split": record["split"],
                        "duration": duration,
                    })

                    speaker_counts[speaker] += 1

                    if len(rows) % 20 == 0:
                        print(f"Extracted {len(rows)} WAV files")

                    if len(rows) >= TARGET_COUNT:
                        break

    except (tarfile.ReadError, EOFError, OSError) as exc:
        print(f"Archive stream ended: {exc}")

    if not rows:
        raise RuntimeError(
            "No usable audio was extracted. "
            "Check archive access and file structure."
        )

    rows.sort(key=lambda row: row["id"])

    debug_rows = build_debug_subset(rows)

    manifest = DATA_DIR / "manifest.csv"
    debug_manifest = DATA_DIR / "debug_manifest.csv"

    write_manifest(manifest, rows)
    write_manifest(debug_manifest, debug_rows)

    print("\nPreparation complete")
    print(f"Total samples: {len(rows)}")
    print(f"Debug samples: {len(debug_rows)}")
    print(f"Speakers: {len(speaker_counts)}")
    print(f"Language tags: {dict(Counter(r['TAG'] for r in rows))}")
    print(f"Splits: {dict(Counter(r['split'] for r in rows))}")
    print(f"Manifest: {manifest}")
    print(f"Debug manifest: {debug_manifest}")


if __name__ == "__main__":
    main()
