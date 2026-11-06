# Project outline

Full working version: the shared "395_FinalProject Outline" document.

## Proposal (1 paragraph) -- TODO: write in our own words

Problem, dataset, planned approach.

## Goal

Off-the-shelf ASR models make many errors on Mandarin-English code-switched speech,
especially on audio from a low-cost microphone. We fine-tune a pretrained speech encoder
with a CTC head to lower its mixed error rate (MER) on code-switched conversation recorded
on an ESP32, and analyze whether errors concentrate where speakers switch languages.

## Data

- Public corpus (TBD; ASCEND is the current candidate): Hugging Face `CAiRE/ASCEND`,
  CC BY-SA 4.0 -- 10.62 h of spontaneous Mandarin-English conversation, pre-split
  train/val/test. Train split for fine-tuning, test split as a public benchmark.
- Our recordings: 3 conversations x 5-10 min among international students, recorded on
  one ESP32 board. 1 min of room tone per session for noise augmentation. A phone/video
  recording only helps transcription. Split by conversation: 2 train, 1 test.
- Noise augmentation: room tone at a set SNR + simulated reverb (pyroomacoustics),
  training data only.

## Model

- Pretrained speech encoder (wav2vec 2.0 family) + new linear output layer, trained with
  CTC loss (existing implementation: `Wav2Vec2ForCTC` / `torch.nn.CTCLoss`).
- Vocabulary: Chinese characters + English letters + word delimiter + blank.
- Selection criteria: pretrained on Mandarin and English speech; not trained on our public
  corpus; trainable on an 8 GB GPU; available on Hugging Face.
- The CTC head starts untrained, so the zero-shot baseline is a separate off-the-shelf
  ASR model (e.g., Whisper), used as a reference.

## Experiments

Zero-shot reference -> A (clean public) -> A-noise (+ noise) -> B (+ our 2 conversations).
Test sets: public test split, our held-out conversation (ESP32). Metric: MER.

## Analysis

- Switch-point error rate: MER near a Mandarin/English switch vs. elsewhere.
- CTC frame-probability plot (diagnostic).
- If time allows: noise curve (MER at 20 / 10 / 0 dB SNR).

## Roles -- TODO

- Mia:
- Charlie:
