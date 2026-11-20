"""Off-the-shelf ASR model (no fine-tuning): Whisper zero-shot.

Two uses:
1. Zero-shot reference: run it on the test sets and report MER
   (scripts/run_whisper_zeroshot.py).
2. Draft transcripts of our recordings, which we then correct by hand.

Two decoding variants (decided 2026-11-11):
  "raw"       -- Whisper as-is. Compare with the CS-Dialogue paper (Whisper large-v2: 15.29%).
  "nodigits"  -- every vocabulary token that contains a digit is banned during generation,
                 so Whisper has to write numbers as words ("七百一十一", "SEVEN"), the same
                 way the CS-Dialogue references do. Our CTC model cannot output digits
                 either, so this is the fairer reference line for it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16_000          # Whisper (and CS-Dialogue) use 16 kHz audio
MAX_SECONDS = 30.0            # Whisper hears 30 s at a time; longer audio is done window by window


# ---- small helpers -------------------------------------------------------------------

def has_digit(text: str) -> bool:
    """True if `text` contains any digit character (0-9, full-width ０-９, ...)."""
    return any(ch.isdigit() for ch in text)


def pick_device(requested: str = "auto") -> str:
    """'auto' -> cuda if available, else mps (Apple GPU), else cpu."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(path: str | Path) -> np.ndarray:
    """Read a WAV file as mono float32 at 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:                          # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:                        # CS-Dialogue is already 16 kHz; our ESP32 may not be
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def find_digit_token_ids(tokenizer) -> list[int]:
    """IDs of every ordinary vocabulary token whose text contains a digit.

    Whisper's vocabulary has tokens like "7", "11", " 2025", "x2". Banning all of them
    means no digit can appear in the output. Special tokens (<|zh|>, <|0.00|> timestamps,
    ...) are skipped -- the timestamp tokens contain digits but are not text.
    """
    special = set(tokenizer.all_special_ids)
    ids = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None or (piece.startswith("<|") and piece.endswith("|>")):
            continue
        if has_digit(tokenizer.decode([token_id])):
            ids.append(token_id)
    return ids


# ---- the transcriber -------------------------------------------------------------------

@dataclass
class WhisperTranscriber:
    """Load Whisper once, then call .transcribe(list_of_wav_paths, variant=...)."""

    model_name: str = "openai/whisper-large-v2"
    device: str = "auto"
    dtype: str = "auto"              # "auto": float16 on cuda, float32 on mps/cpu
    language: str | None = None      # None = Whisper detects the language per utterance
    num_beams: int = 1               # 1 = greedy decoding
    _digit_ids: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.device = pick_device(self.device)
        if self.dtype == "auto":
            self.dtype = "float16" if self.device == "cuda" else "float32"
        torch_dtype = getattr(torch, self.dtype)

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, dtype=torch_dtype
        ).to(self.device).eval()

        # Whisper already suppresses some tokens by default (♪, brackets ...). Keep those.
        self._default_suppress = list(self.model.generation_config.suppress_tokens or [])
        self._digit_ids = find_digit_token_ids(self.processor.tokenizer)

    @property
    def n_digit_tokens(self) -> int:
        return len(self._digit_ids)

    @torch.inference_mode()
    def transcribe_batch(self, audios: list[np.ndarray], variant: str = "raw") -> list[str]:
        """Transcribe a batch of 16 kHz waveforms. variant = "raw" or "nodigits".

        Clips <= 30 s and clips > 30 s are decoded separately, so a short clip's result
        never depends on whether a long clip happened to be in the same batch.
        """
        if variant not in ("raw", "nodigits"):
            raise ValueError(f"unknown variant: {variant}")

        limit = MAX_SECONDS * SAMPLE_RATE
        short_idx = [i for i, a in enumerate(audios) if len(a) <= limit]
        long_idx = [i for i, a in enumerate(audios) if len(a) > limit]

        out: list[str] = [""] * len(audios)
        if short_idx:
            texts = self._generate([audios[i] for i in short_idx], variant, long_form=False)
            for i, t in zip(short_idx, texts):
                out[i] = t
        if long_idx:
            texts = self._generate([audios[i] for i in long_idx], variant, long_form=True)
            for i, t in zip(long_idx, texts):
                out[i] = t
        return out

    def _generate(self, audios: list[np.ndarray], variant: str, long_form: bool) -> list[str]:
        # waveform -> log-Mel spectrogram, 100 frames per second.
        if long_form:
            # Clips > 30 s: do NOT cut them. Pad to the longest clip and pass a mask;
            # Whisper then transcribes 30 s windows one after another ("long-form").
            inputs = self.processor.feature_extractor(
                audios, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                truncation=False, padding="longest", return_attention_mask=True,
            )
            extra = {"attention_mask": inputs.attention_mask.to(self.device),
                     "return_timestamps": True}   # long-form needs timestamps to move the window
        else:
            # Clips <= 30 s: pad each to exactly 30 s, shape (batch, 80, 3000)
            inputs = self.processor.feature_extractor(
                audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            extra = {}
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)

        suppress = self._default_suppress
        if variant == "nodigits":
            # digit IDs go FIRST and the default list keeps its order at the end:
            # Whisper's long-form code reads suppress_tokens[-2] as a special token.
            default = set(suppress)
            suppress = [i for i in self._digit_ids if i not in default] + suppress

        gen_cfg = copy.deepcopy(self.model.generation_config)
        gen_cfg.suppress_tokens = suppress
        gen_cfg.num_beams = self.num_beams
        generated = self.model.generate(
            features,
            generation_config=gen_cfg,
            task="transcribe",            # never "translate" -- we want the words as spoken
            language=self.language,       # None = auto-detect per utterance
            **extra,
        )
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def transcribe(self, paths: list[str | Path], variant: str = "raw",
                   batch_size: int = 8, progress: bool = True) -> list[str]:
        """Transcribe WAV files in batches; returns one string per file, in order."""
        from tqdm import tqdm

        out: list[str] = []
        starts = range(0, len(paths), batch_size)
        for start in tqdm(starts, desc=variant, disable=not progress):
            batch = [load_audio(p) for p in paths[start:start + batch_size]]
            out.extend(self.transcribe_batch(batch, variant=variant))
        return out
