"""Evaluation: mixed error rate (MER).

MER = (S + D + I) / N
  S = substitutions, D = deletions, I = insertions,
  N = number of tokens in the reference,
  where a token is one Chinese character or one English word (see normalize.py).

Scores are corpus-level: errors and N are summed over ALL utterances first, then divided
(not an average of per-utterance MERs), which is how the CS-Dialogue paper reports MER.

TODO (Step 4): greedy CTC decoding -- take the most likely token per frame, merge
repeats, drop blanks.
"""

from collections import defaultdict

import jiwer

from codeswitch_asr.normalize import normalize


def compute_mer(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level MER of hypotheses `hyps` against references `refs` (raw, un-normalized text).

    Returns a dict with mer, the error counts, n_ref_tokens, n_utts, and n_empty_refs
    (utterances skipped because the reference is empty after normalization,
    e.g. a reference that was only "<FIL/>").
    """
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")

    # 1. normalize both sides with the same rules
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]

    # 2. drop pairs whose reference became empty (MER is undefined when N = 0)
    kept = [(r, h) for r, h in pairs if r]
    n_empty = len(pairs) - len(kept)
    if not kept:
        raise ValueError("every reference is empty after normalization")
    norm_refs, norm_hyps = zip(*kept)

    # 3. jiwer aligns each pair token by token (tokens = space-separated) and sums the counts
    out = jiwer.process_words(list(norm_refs), list(norm_hyps))
    s, d, i = out.substitutions, out.deletions, out.insertions
    n = s + d + out.hits                       # reference tokens = substituted + deleted + correct

    return {
        "mer": (s + d + i) / n,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": out.hits,
        "n_ref_tokens": n,
        "n_utts": len(kept),
        "n_empty_refs": n_empty,
    }


def compute_mer_by_group(refs: list[str], hyps: list[str], groups: list[str]) -> dict:
    """MER for every group separately, plus "ALL".

    Example: groups = the CS-Dialogue TAG of each utterance ("<CN>", "<EN>", "<MIX>"),
    so result["<MIX>"]["mer"] is the MIX-only test MER.
    """
    if not (len(refs) == len(hyps) == len(groups)):
        raise ValueError("refs, hyps and groups must have the same length")

    by_group: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for r, h, g in zip(refs, hyps, groups):
        by_group[g][0].append(r)
        by_group[g][1].append(h)

    results = {"ALL": compute_mer(refs, hyps)}
    for g, (rs, hs) in sorted(by_group.items()):
        results[g] = compute_mer(rs, hs)
    return results
