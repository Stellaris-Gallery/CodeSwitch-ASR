"""Step 3: Whisper zero-shot MER on CS-Dialogue (no training).

Runs Whisper twice on every utterance in a manifest CSV -- "raw" and "nodigits"
(see src/codeswitch_asr/transcribe.py) -- and reports corpus-level MER for
ALL / <CN> / <EN> / <MIX>.

Pilot (200 utterances, Mia's Mac):
    uv run python scripts/run_whisper_zeroshot.py \
        --manifest data/cs_dialogue_pilot/manifest.csv

Full test set (6,257 utterances, workstation):
    uv run python scripts/run_whisper_zeroshot.py \
        --manifest "%CS_DIALOGUE_ROOT%/exports/test_full_6257.csv" \
        --audio-root "%CS_DIALOGUE_ROOT%" --batch-size 32

Manifest columns (from scripts/export_cs_dialogue_test.py):
    id, audio_path, text, TAG, speaker, split, duration
audio_path is relative to --audio-root (default: the folder the manifest is in).

Outputs (in --out-dir, which is git-ignored under outputs/):
    hyps.csv      one row per utterance: reference + both hypotheses
    summary.json  MER per variant per TAG, digit counts, run settings
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from codeswitch_asr.evaluate import compute_mer_by_group
from codeswitch_asr.transcribe import MAX_SECONDS, WhisperTranscriber, has_digit

VARIANTS = ("raw", "nodigits")
GROUP_ORDER = ("ALL", "<CN>", "<EN>", "<MIX>")


def read_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def print_table(results: dict[str, dict]) -> None:
    print(f"\n{'variant':<10}" + "".join(f"{g:>10}" for g in GROUP_ORDER))
    for variant, by_group in results.items():
        cells = "".join(
            f"{by_group[g]['mer'] * 100:>9.2f}%" if g in by_group else f"{'-':>10}"
            for g in GROUP_ORDER
        )
        print(f"{variant:<10}{cells}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="folder that audio_path is relative to (default: manifest's folder)")
    ap.add_argument("--model", default="openai/whisper-large-v2")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--language", default=None,
                    help="force a language, e.g. zh (default: Whisper auto-detects per utterance)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--device", default="auto", help="auto / mps / cuda / cpu")
    ap.add_argument("--dtype", default="auto", help="auto / float32 / float16")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (quick test)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true", help="also log the MER table to W&B")
    args = ap.parse_args()

    rows = read_manifest(args.manifest, args.limit)
    audio_root = args.audio_root or args.manifest.parent
    paths = [audio_root / r["audio_path"] for r in rows]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files missing, e.g. {missing[0]}")

    refs = [r["text"] for r in rows]
    tags = [r["TAG"] for r in rows]
    too_long = sum(float(r.get("duration") or 0) > MAX_SECONDS for r in rows)
    print(f"{len(rows)} utterances  TAGs: {dict(Counter(tags))}")
    if too_long:
        print(f"{too_long} utterances are longer than {MAX_SECONDS:.0f}s -> transcribed in 30 s windows (long-form)")

    model_short = args.model.split("/")[-1]
    lang = args.language or "auto"
    out_dir = args.out_dir or Path("outputs/zeroshot") / f"{model_short}_lang-{lang}_{args.manifest.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    asr = WhisperTranscriber(model_name=args.model, device=args.device, dtype=args.dtype,
                             language=args.language, num_beams=args.num_beams)
    print(f"model={args.model} device={asr.device} dtype={asr.dtype} language={lang} "
          f"digit tokens banned in 'nodigits': {asr.n_digit_tokens}")

    hyps: dict[str, list[str]] = {}
    seconds: dict[str, float] = {}
    for variant in args.variants:
        t0 = time.time()
        hyps[variant] = asr.transcribe(paths, variant=variant, batch_size=args.batch_size)
        seconds[variant] = round(time.time() - t0, 1)

    # ---- per-utterance CSV ------------------------------------------------------------
    with (out_dir / "hyps.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "TAG", "speaker", "duration", "ref"] + [f"hyp_{v}" for v in args.variants]
        if "raw" in hyps:
            fields.append("raw_has_digit")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            line = {"id": r["id"], "TAG": r["TAG"], "speaker": r.get("speaker", ""),
                    "duration": r.get("duration", ""), "ref": r["text"]}
            for v in args.variants:
                line[f"hyp_{v}"] = hyps[v][i]
            if "raw" in hyps:
                line["raw_has_digit"] = int(has_digit(hyps["raw"][i]))
            w.writerow(line)

    # ---- MER --------------------------------------------------------------------------
    results = {v: compute_mer_by_group(refs, hyps[v], tags) for v in args.variants}
    digit_counts = {
        v: {"ALL": sum(has_digit(h) for h in hyps[v]),
            **dict(Counter(t for t, h in zip(tags, hyps[v]) if has_digit(h)))}
        for v in args.variants
    }

    print_table(results)
    for v, counts in digit_counts.items():
        print(f"utterances whose '{v}' output contains a digit: {counts}")

    summary = {
        "model": args.model, "manifest": str(args.manifest), "n_utts": len(rows),
        "language": lang, "num_beams": args.num_beams, "device": asr.device, "dtype": asr.dtype,
        "n_digit_tokens_banned": asr.n_digit_tokens, "seconds": seconds,
        "utts_over_30s": too_long, "mer": results, "hyps_with_digits": digit_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved {out_dir / 'hyps.csv'} and {out_dir / 'summary.json'}")

    if args.wandb:
        import wandb
        run = wandb.init(project="codeswitch-asr", job_type="eval_zeroshot",
                         name=out_dir.name, config={k: v for k, v in summary.items() if k != "mer"})
        table = wandb.Table(columns=["variant", *GROUP_ORDER])
        for v, by_group in results.items():
            table.add_data(v, *[by_group.get(g, {}).get("mer") for g in GROUP_ORDER])
            for g, res in by_group.items():
                run.summary[f"{v}/mer_{g.strip('<>')}"] = res["mer"]
        run.log({"mer_table": table})
        run.finish()


if __name__ == "__main__":
    main()
