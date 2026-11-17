"""Local CS-Dialogue audio/label loading for character-level CTC training.

The train vocabulary must be generated ONLY from kept train utterances by
scripts/build_cs_dialogue_ctc.py. This module never changes that vocabulary.

Train manifest: <CS_DIALOGUE_ROOT>/processed/train_ctc_manifest.csv
Dev/test: loaded from the official short_wav Kaldi indexes, without filtering.
"""

from __future__ import annotations

import csv
import json
import os
import re
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset

from codeswitch_asr.normalize import tokenize

EXPECTED = {"train": 25_568, "dev": 6_186, "test": 6_257}
EXPECTED_RAW_TRAIN = 26_239
SAMPLE_RATE = 16_000
_METADATA_RE = re.compile(
    r"(?P<wav>[^\s]+\.wav)\s+(?P<tag><[A-Za-z][A-Za-z0-9_-]*>)\s+(?P<transcript>.*)$"
)
_TAGS = {"<CN>", "<EN>", "<MIX>"}
FIELDNAMES = [
    "id", "audio_path", "text", "TAG", "speaker", "split", "duration",
    "normalized_text", "ctc_text",
]


@dataclass(frozen=True)
class ASRRecord:
    """An utterance and its normalized character-level CTC target."""

    id: str
    audio_path: Path
    text: str
    tag: str
    speaker: str
    split: str
    duration: float
    ctc_text: str


def ctc_text_from_transcript(text: str) -> str:
    """Match train preparation: MER tokens separated by CTC's '|' character."""
    return "|".join(tokenize(text))


def read_kaldi(path: Path) -> dict[str, str]:
    """Read the official Kaldi ID/value index; reject duplicate and malformed IDs."""
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Malformed Kaldi index {path}:{line_number}")
            key, value = parts
            if key in result:
                raise ValueError(f"Duplicate Kaldi ID {key}: {path}")
            result[key] = value
    return result


def load_official_metadata(dataset_root: Path) -> dict[str, tuple[str, str]]:
    """Read official speaker/TAG assignments; never infer TAGs from transcripts."""
    path = (
        dataset_root / "data" / "index" / "total_information"
        / "Information_Index.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing official speaker metadata: {path}")
    result: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            stripped = line.strip()
            speaker_match = re.match(r"^(\d+)\s+", stripped)
            entry = _METADATA_RE.search(stripped)
            if not speaker_match or not entry:
                continue
            key = Path(entry.group("wav").replace("\\", "/")).stem
            speaker, tag = speaker_match.group(1), entry.group("tag")
            if key in result:
                raise ValueError(f"Duplicate official metadata ID: {key}")
            result[key] = (speaker, tag)
    if not result:
        raise ValueError(f"No usable official metadata in {path}")
    return result


def _audio_index(dataset_root: Path) -> dict[str, Path]:
    audio_root = dataset_root / "audio" / "short_wav"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Extracted short_wav missing: {audio_root}")
    lookup: dict[str, Path] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            if path.stem in lookup:
                raise ValueError(f"Duplicate WAV ID {path.stem}: {path}")
            lookup[path.stem] = path
    return lookup


def _safe_audio_path(dataset_root: Path, relative_path: str) -> Path:
    """Reject missing files and paths that escape the dataset directory."""
    path = (dataset_root / relative_path.replace("/", os.sep)).resolve()
    if not path.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"Audio path escapes dataset root: {relative_path}")
    return path


def load_train_manifest(dataset_root: Path, *, check_files: bool = False) -> list[ASRRecord]:
    """Load generated, already-filtered train manifest (does NOT re-filter)."""
    dataset_root = Path(dataset_root).resolve()
    path = dataset_root / "processed" / "train_ctc_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Build train CTC manifest first: {path}")
    records: list[ASRRecord] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = set(FIELDNAMES)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = row["id"]
            if key in seen:
                raise ValueError(f"Duplicate train utterance ID {key}")
            seen.add(key)
            if row["split"] != "train":
                raise ValueError(f"Non-train row found in train manifest: {key}")
            if row["TAG"] not in _TAGS:
                raise ValueError(f"Unexpected TAG {row['TAG']} for {key}")
            if "**" in row["text"]:
                raise ValueError(f"Unintelligible train row was not filtered: {key}")
            expected = ctc_text_from_transcript(row["text"])
            if row["ctc_text"] != expected or not expected:
                raise ValueError(f"CTC normalization mismatch or empty target: {key}")
            audio_path = _safe_audio_path(dataset_root, row["audio_path"])
            if check_files and not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            records.append(ASRRecord(
                id=key, audio_path=audio_path, text=row["text"],
                tag=row["TAG"], speaker=row["speaker"], split="train",
                duration=float(row["duration"]), ctc_text=expected,
            ))
    if len(records) != EXPECTED["train"]:
        raise ValueError(f"Expected {EXPECTED['train']} kept train rows, got {len(records)}")
    return records


def load_official_eval_split(dataset_root: Path, split: str) -> list[ASRRecord]:
    """Load EVERY official dev/test utterance; do not filter **, durations or speakers.

    Dev/test OOV characters remain in the reference and map to [UNK] in the
    CTC tokenizer; report the OOV count before training/evaluating.
    """
    if split not in ("dev", "test"):
        raise ValueError("Evaluation split must be 'dev' or 'test'")
    root = Path(dataset_root).resolve()
    idx = root / "data" / "index" / "short_wav" / split
    texts, audio_refs = read_kaldi(idx / "text"), read_kaldi(idx / "wav.scp")
    if set(texts) != set(audio_refs):
        raise ValueError(f"{split} text and wav.scp index IDs differ")
    if len(texts) != EXPECTED[split]:
        raise ValueError(f"Expected {EXPECTED[split]} {split} entries, got {len(texts)}")
    metadata, audio = load_official_metadata(root), _audio_index(root)
    results: list[ASRRecord] = []
    for key, text in sorted(texts.items()):
        if key not in metadata or key not in audio:
            raise ValueError(f"Missing official metadata/audio for {split} ID: {key}")
        speaker, tag = metadata[key]
        if tag not in _TAGS:
            raise ValueError(f"Unexpected TAG for {key}: {tag}")
        path = audio[key]
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"Unexpected WAV format: {path}")
            duration = wav.getnframes() / SAMPLE_RATE
        results.append(ASRRecord(
            id=key, audio_path=path, text=text, tag=tag, speaker=speaker,
            split=split, duration=round(duration, 3),
            ctc_text=ctc_text_from_transcript(text),
        ))
    return results


def write_eval_manifest(rows: list[ASRRecord], dataset_root: Path) -> Path:
    """Save a local-only evaluation manifest. Do not change/drop official rows."""
    if not rows or any(row.split != rows[0].split for row in rows):
        raise ValueError("Expected one non-empty evaluation split")
    if rows[0].split not in ("dev", "test"):
        raise ValueError("Only evaluation manifests can be written")
    root = Path(dataset_root).resolve()
    output = root / "processed" / f"{rows[0].split}_ctc_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row.id, "audio_path": row.audio_path.relative_to(root).as_posix(),
                "text": row.text, "TAG": row.tag, "speaker": row.speaker,
                "split": row.split, "duration": row.duration,
                "normalized_text": " ".join(tokenize(row.text)),
                "ctc_text": row.ctc_text,
            })
    return output


def load_vocab(vocab_file: Path) -> dict[str, int]:
    """Validate the train-only vocabulary and reserved IDs."""
    data = json.loads(Path(vocab_file).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not data:
        raise ValueError("CTC vocabulary is empty or not a JSON object")
    if any(not isinstance(key, str) or not isinstance(value, int) for key, value in data.items()):
        raise ValueError("Vocabulary must map string symbols to integer IDs")
    if set(data.values()) != set(range(len(data))):
        raise ValueError("Vocabulary IDs must be contiguous and unique")
    if len(data) != 2951:
        raise ValueError(f"Expected 2,951 train-only vocab entries, got {len(data)}")
    for special in ("|", "[UNK]", "<blank>"):
        if special not in data:
            raise ValueError(f"Missing required CTC token: {special}")
    return data


def vocab_coverage(records: list[ASRRecord], vocab: dict[str, int]) -> dict[str, Any]:
    """Check per-character OOV coverage; do not alter original transcripts."""
    outside = Counter(char for row in records for char in row.ctc_text if char not in vocab)
    return {
        "rows": len(records),
        "empty_targets": sum(not row.ctc_text for row in records),
        "oov_characters": sum(outside.values()),
        "oov_utterances": sum(any(ch not in vocab for ch in row.ctc_text) for row in records),
        "oov_top_20": outside.most_common(20),
    }


def build_processor(vocab_file: Path, pretrained_model: str = "facebook/wav2vec2-xls-r-300m"):
    """Build local train-only CTC tokenizer and official XLS-R feature extractor.

    Important: bos/eos are explicitly disabled so tokenizer size stays 2,951.
    The `<blank>` token is both CTC blank and Hugging Face padding ID.
    """
    from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

    vocab = load_vocab(vocab_file)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file), unk_token="[UNK]", pad_token="<blank>",
        word_delimiter_token="|", bos_token=None, eos_token=None,
        do_lower_case=False,
    )
    if len(tokenizer) != len(vocab):
        raise ValueError(
            f"Tokenizer unexpectedly added tokens: {len(tokenizer)} vs {len(vocab)}"
        )
    if tokenizer.pad_token_id != vocab["<blank>"]:
        raise ValueError("CTC blank/pad ID mismatch")
    if tokenizer.unk_token_id != vocab["[UNK]"]:
        raise ValueError("CTC unknown token ID mismatch")
    features = AutoFeatureExtractor.from_pretrained(pretrained_model)
    if getattr(features, "sampling_rate", None) != SAMPLE_RATE:
        raise ValueError("XLS-R feature extractor did not request 16-kHz audio")
    return Wav2Vec2Processor(feature_extractor=features, tokenizer=tokenizer)


class CSDialogueCTCDataset(Dataset):
    """Lazily read WAVs and create input_values/CTC labels on demand."""

    def __init__(self, records: list[ASRRecord], processor, *, allow_oov: bool = False):
        self.records = records
        self.processor = processor
        vocab = processor.tokenizer.get_vocab()
        if not allow_oov:
            for row in records:
                if any(char not in vocab for char in row.ctc_text):
                    raise ValueError(f"OOV token in strict train sample: {row.id}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        audio, sample_rate = sf.read(str(row.audio_path), dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Expected mono 16-kHz WAV: {row.audio_path}")
        values = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_attention_mask=True
        ).input_values[0]
        labels = self.processor.tokenizer(
            row.ctc_text, add_special_tokens=False
        ).input_ids
        if not labels and row.split == "train":
            raise ValueError(f"Empty CTC target in train: {row.id}")
        if self.processor.tokenizer.pad_token_id in labels:
            raise ValueError(f"CTC blank unexpectedly appears in label target: {row.id}")
        return {
            "id": row.id, "input_values": values,
            "labels": labels, "text": row.text, "ctc_text": row.ctc_text,
        }


class CTCDataCollator:
    """Dynamically pad audio and mask padded CTC labels with -100."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_batch = self.processor.feature_extractor.pad(
            [{"input_values": item["input_values"]} for item in features],
            padding=True, return_attention_mask=True, return_tensors="pt",
        )
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features],
            padding=True, return_tensors="pt",
        )
        audio_batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        return dict(audio_batch)
