"""Offline-only unit tests for core CTC data conventions."""

import csv
import json
from pathlib import Path

import pytest

from codeswitch_asr.data import (
    ctc_text_from_transcript, load_train_manifest, load_vocab, read_kaldi,
    vocab_coverage, ASRRecord,
)


def test_ctc_text_uses_shared_normalization():
    assert ctc_text_from_transcript("我用 I B M 的 LAPTOP。") == "我|用|IBM|的|LAPTOP"
    assert ctc_text_from_transcript("<FIL/>你好 don't") == "你|好|DON'T"


def test_kaldi_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "text"
    p.write_text("id1 hello\nid1 again\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_kaldi(p)


def test_vocabulary_special_ids_and_size(tmp_path):
    path = tmp_path / "vocab.json"
    vocab = {"|": 0, **{f"C{i}": i + 1 for i in range(2948)}, "[UNK]": 2949, "<blank>": 2950}
    path.write_text(json.dumps(vocab), encoding="utf-8")
    assert len(load_vocab(path)) == 2951
    vocab["<blank>"] = 9999
    path.write_text(json.dumps(vocab), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_vocab(path)


def test_oov_coverage_counts_each_character(tmp_path):
    sample = ASRRecord("abc", tmp_path / "a.wav", "", "<CN>", "0001", "dev", 1.0, "你|好|XYZ")
    stats = vocab_coverage([sample], {"你": 0, "好": 1, "|": 2, "X": 3})
    assert stats["oov_characters"] == 2
    assert stats["oov_utterances"] == 1


def test_train_manifest_rejects_wrong_split(tmp_path):
    p = tmp_path / "processed" / "train_ctc_manifest.csv"
    p.parent.mkdir()
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "audio_path", "text", "TAG", "speaker", "split",
            "duration", "normalized_text", "ctc_text",
        ])
        writer.writeheader()
        writer.writerow(dict(id="abc", audio_path="audio/abc.wav", text="你好", TAG="<CN>", speaker="0001", split="test", duration="1", normalized_text="你 好", ctc_text="你|好"))
    with pytest.raises(ValueError, match="Non-train"):
        load_train_manifest(tmp_path)
