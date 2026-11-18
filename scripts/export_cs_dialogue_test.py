"""Export reproducible CS-Dialogue test manifests and a 200-WAV Whisper pilot bundle.

Run from the CodeSwitch-ASR repository with:
    uv run python scripts/export_cs_dialogue_test.py --dataset-root "D:\\Deeplearning Final Project\\Datasets\\CS-Dialogue"

This script writes only to DATASET_ROOT/exports; it never modifies original WAVs.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import wave
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

EXPECTED_SPLITS = {"train": 26239, "dev": 6186, "test": 6257}
TAGS = ("<CN>", "<MIX>", "<EN>")
FIELDS = ["id", "audio_path", "text", "TAG", "speaker", "split", "duration"]
META_LINE = re.compile(
    r"(?P<wav>[^\s]+\.wav)\s+(?P<tag><[A-Za-z][A-Za-z0-9_-]*>)\s+(?P<script>.*)$"
)


def read_kaldi(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official index: {path}")
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed {path}:{line_number}")
            identifier, value = parts
            if identifier in result:
                raise ValueError(f"Duplicate ID {identifier} in {path}")
            result[identifier] = value
    return result


def load_official_index(dataset_root: Path) -> dict[str, dict[str, str]]:
    base = dataset_root / "data" / "index" / "short_wav"
    records: dict[str, dict[str, str]] = {}
    for split, expected_count in EXPECTED_SPLITS.items():
        texts = read_kaldi(base / split / "text")
        scp = read_kaldi(base / split / "wav.scp")
        if set(texts) != set(scp):
            raise ValueError(
                f"{split}: text/wav.scp IDs differ: "
                f"text-only={len(set(texts)-set(scp))}, scp-only={len(set(scp)-set(texts))}"
            )
        if len(texts) != expected_count:
            raise ValueError(f"{split}: expected {expected_count}, found {len(texts)}")
        for identifier, transcript in texts.items():
            if identifier in records:
                raise ValueError(f"ID {identifier} appears in multiple splits")
            records[identifier] = {
                "id": identifier,
                "text": transcript,
                "split": split,
                "scp": scp[identifier],
            }
        print(f"Official {split}: {len(texts):,} indexed utterances")
    return records


def load_official_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    path = (
        dataset_root / "data" / "index" / "total_information" / "Information_Index.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing official TAG/speaker metadata: {path}\n"
            "Download with: uvx hf download BAAI/CS-Dialogue --repo-type dataset "
            '--include "data/index/total_information/*" --local-dir "<dataset-root>"'
        )
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("Speaker ID"):
                continue
            speaker_match = re.match(r"^(\d+)\s+", line)
            entry = META_LINE.search(line)
            if speaker_match is None or entry is None:
                continue
            identifier = Path(entry.group("wav").replace("\\", "/")).stem
            metadata = {"speaker": speaker_match.group(1), "TAG": entry.group("tag")}
            if identifier in result and result[identifier] != metadata:
                raise ValueError(f"Conflicting metadata: {identifier}")
            result[identifier] = metadata
    print(f"Official metadata IDs: {len(result):,}")
    return result


def index_audio(dataset_root: Path) -> tuple[dict[str, Path], list[tuple[str, str, str]], int]:
    audio_root = dataset_root / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing extracted audio directory: {audio_root}")
    by_id: dict[str, Path] = {}
    duplicates: list[tuple[str, str, str]] = []
    wav_count = 0
    for wav in audio_root.rglob("*"):
        if not wav.is_file() or wav.suffix.lower() != ".wav":
            continue
        wav_count += 1
        identifier = wav.stem
        if identifier in by_id:
            duplicates.append((identifier, str(by_id[identifier]), str(wav)))
        else:
            by_id[identifier] = wav
    print(f"Extracted WAV files: {wav_count:,} ({len(by_id):,} unique filename IDs)")
    return by_id, duplicates, wav_count


def duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            if (rate, channels, sample_width) != (16000, 1, 2):
                raise ValueError(
                    f"Unexpected WAV format for {path.name}: "
                    f"rate={rate}, channels={channels}, sample_width={sample_width}"
                )
            return round(audio.getnframes() / rate, 3)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Cannot read WAV: {path}") from exc


def choose_balanced(rows: list[dict[str, str | float]], count: int, seed: int):
    """Balance TAGs and rotate speakers so multiple speakers are represented."""
    quotas = {tag: count // len(TAGS) for tag in TAGS}
    for tag in TAGS[: count % len(TAGS)]:
        quotas[tag] += 1
    rng = random.Random(seed)
    chosen: list[dict[str, str | float]] = []
    for tag in TAGS:
        groups: dict[str, list[dict[str, str | float]]] = defaultdict(list)
        for row in rows:
            if row["TAG"] == tag:
                groups[str(row["speaker"])].append(row)
        for items in groups.values():
            items.sort(key=lambda item: str(item["id"]))
            rng.shuffle(items)
        speakers = sorted(groups)
        rng.shuffle(speakers)
        buckets = {speaker: deque(groups[speaker]) for speaker in speakers}
        tag_chosen = []
        while len(tag_chosen) < quotas[tag]:
            progressed = False
            for speaker in speakers:
                if buckets[speaker] and len(tag_chosen) < quotas[tag]:
                    tag_chosen.append(buckets[speaker].popleft())
                    progressed = True
            if not progressed:
                raise ValueError(
                    f"Insufficient {tag} examples: need {quotas[tag]}, got {len(tag_chosen)}"
                )
        chosen.extend(tag_chosen)
    chosen.sort(key=lambda row: str(row["id"]))
    if len(chosen) != count or len({row["id"] for row in chosen}) != count:
        raise AssertionError("Pilot subset selection yielded incorrect/duplicate IDs")
    return chosen


def csv_bytes(rows: list[dict[str, str | float]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=395)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be a positive integer")
    dataset_root: Path = args.dataset_root.resolve()
    output = dataset_root / "exports"
    official = load_official_index(dataset_root)
    metadata = load_official_metadata(dataset_root)
    wavs, duplicates, wav_count = index_audio(dataset_root)
    official_ids = set(official)
    ambiguous = sorted({item[0] for item in duplicates} & official_ids)
    if ambiguous:
        raise RuntimeError(
            f"Duplicate audio filename IDs are referenced by the official index: {ambiguous[:10]}"
        )
    missing_audio = sorted(official_ids - set(wavs))
    missing_meta = sorted(official_ids - set(metadata))
    if missing_audio or missing_meta:
        raise RuntimeError(
            f"Missing audio for {len(missing_audio)} official IDs, "
            f"missing metadata for {len(missing_meta)} official IDs. "
            f"Examples: audio={missing_audio[:5]}, metadata={missing_meta[:5]}"
        )
    extra_audio = sorted(set(wavs) - official_ids)
    output.mkdir(parents=True, exist_ok=True)
    (output / "unindexed_audio_ids.txt").write_text(
        "\n".join(extra_audio) + ("\n" if extra_audio else ""), encoding="utf-8"
    )
    print(
        f"Audit: {len(extra_audio):,} unindexed audio IDs; "
        f"{len(duplicates):,} duplicated filename IDs; "
        f"{wav_count-len(wavs):,} extra physical files with repeated IDs"
    )

    # Keep all official test examples, regardless of length or speaker frequency.
    test_rows: list[dict[str, str | float]] = []
    for identifier in sorted(official_ids):
        entry = official[identifier]
        if entry["split"] != "test":
            continue
        tag = metadata[identifier]["TAG"]
        if tag not in TAGS:
            raise ValueError(f"Unknown official TAG for {identifier}: {tag}")
        path = wavs[identifier]
        test_rows.append(
            {
                "id": identifier,
                "audio_path": path.relative_to(dataset_root).as_posix(),
                "text": entry["text"],
                "TAG": tag,
                "speaker": metadata[identifier]["speaker"],
                "split": "test",
                "duration": duration_seconds(path),
            }
        )
    if len(test_rows) != EXPECTED_SPLITS["test"]:
        raise AssertionError("The full test manifest must contain exactly 6,257 rows")
    full_csv = output / "test_full_6257.csv"
    full_csv.write_bytes(csv_bytes(test_rows))

    pilot = choose_balanced(test_rows, count=args.count, seed=args.seed)
    pilot_rows = [{**row, "audio_path": f"audio/{row['id']}.wav"} for row in pilot]
    pilot_csv = output / f"test_pilot_{args.count}.csv"
    pilot_csv.write_bytes(csv_bytes(pilot_rows))
    archive = output / f"test_pilot_{args.count}.zip"
    readme = (
        "CS-Dialogue: Whisper zero-shot test pilot\n"
        f"Official test split: {len(test_rows)} utterances; pilot: {len(pilot)} utterances.\n"
        f"Seed: {args.seed}. Split: test only. Balanced TAGs, multiple speakers.\n"
        "Use manifest.csv; audio_path values are relative to this ZIP root.\n"
        "Text fields are RAW official split-index transcripts (do not pre-normalize here).\n"
        "For MER evaluation, use the project's shared normalize/evaluate functions.\n"
        "Dataset: BAAI/CS-Dialogue (CC BY-NC-SA 4.0); research-only sharing.\n"
        "Do not publish these WAVs or distribute beyond the research group.\n"
    )
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("manifest.csv", csv_bytes(pilot_rows))
        bundle.writestr("README.txt", readme.encode("utf-8"))
        for row in pilot:
            bundle.write(wavs[str(row["id"])], arcname=f"audio/{row['id']}.wav")
    with zipfile.ZipFile(archive) as bundle:
        wav_members = [name for name in bundle.namelist() if name.endswith(".wav")]
        if len(wav_members) != args.count or bundle.testzip() is not None:
            raise RuntimeError("Pilot ZIP validation failed")

    print("\n=== COMPLETE ===")
    print("Full test TAG counts:", dict(Counter(row["TAG"] for row in test_rows)))
    print("Pilot TAG counts:", dict(Counter(row["TAG"] for row in pilot)))
    print("Pilot unique speakers:", len({row["speaker"] for row in pilot}))
    print("Full test manifest:", full_csv)
    print("200-item manifest:", pilot_csv)
    print("Share with Mia:", archive)
    print("Unindexed-audio audit:", output / "unindexed_audio_ids.txt")
    print("No original audio or Git-tracked files were modified.")


if __name__ == "__main__":
    main()
