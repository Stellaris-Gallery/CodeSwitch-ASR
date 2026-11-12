"""Tests for MER. Each expected number was worked out by hand (see comments)."""

import pytest

from codeswitch_asr.evaluate import compute_mer, compute_mer_by_group


def test_perfect_after_normalization():
    # Different writing styles, same words -> 0 errors.
    r = compute_mer(["这个 PROJECT 的 DEADLINE 是明天<FIL/>。"], ["這個project的deadline是明天。"])
    assert r["mer"] == 0
    assert r["n_ref_tokens"] == 8


def test_digits_count_as_errors():
    # ref: 我 住 七 幺 幺 (5 tokens); hyp: 我 住 711
    # 711 replaces 七 (1 substitution), 幺 幺 missing (2 deletions) -> 3 / 5
    r = compute_mer(["我住七幺幺"], ["我住711"])
    assert r["substitutions"] == 1 and r["deletions"] == 2 and r["insertions"] == 0
    assert r["mer"] == pytest.approx(3 / 5)


def test_insertion():
    # ref: 我 们 走 吧 (4); hyp: 我 们 走 吧 OK -> 1 insertion -> 1 / 4
    r = compute_mer(["我们走吧"], ["我们走吧 OK"])
    assert r["insertions"] == 1
    assert r["mer"] == pytest.approx(1 / 4)


def test_corpus_level_not_average():
    # utt 1: 1 error / 2 tokens (50%); utt 2: 0 errors / 8 tokens (0%)
    # corpus-level = (1 + 0) / (2 + 8) = 10%, NOT the average (50% + 0%) / 2 = 25%
    refs = ["你好", "我们今天去吃火锅"]
    hyps = ["你号", "我们今天去吃火锅"]
    r = compute_mer(refs, hyps)
    assert r["mer"] == pytest.approx(1 / 10)


def test_empty_reference_skipped():
    r = compute_mer(["<FIL/>", "好的"], ["嗯", "好的"])
    assert r["n_empty_refs"] == 1
    assert r["n_utts"] == 1
    assert r["mer"] == 0


def test_by_group():
    refs = ["你好", "HELLO WORLD", "我用 I B M"]
    hyps = ["你好", "HELLO WORD", "我用IBM"]
    tags = ["<CN>", "<EN>", "<MIX>"]
    r = compute_mer_by_group(refs, hyps, tags)
    assert r["<CN>"]["mer"] == 0
    assert r["<EN>"]["mer"] == pytest.approx(1 / 2)       # WORD vs WORLD
    assert r["<MIX>"]["mer"] == 0
    assert r["ALL"]["mer"] == pytest.approx(1 / 7)        # 1 error / (2 + 2 + 3) tokens


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_mer(["a", "b"], ["a"])
