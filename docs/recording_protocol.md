# Recording protocol -- TODO: finalize before the first session

## Before recording
- Every participant signs the consent form (stored privately, never in this repo).
  The form must mention both audio and video recording.
- Confirm with Prof. Walsh whether IRB review is needed.

## Setup
- One ESP32 board in the middle of the table; note its position and settings
  (sample rate >= 16 kHz).
- One phone (audio or video) near the speakers -- used ONLY to help transcription,
  not part of the experiments.

## Each session
1. Start the ESP32 and the phone.
2. Record 1 min of room tone (empty room, nobody talking) -- used for noise augmentation.
3. One loud clap heard by both devices (used to align them).
4. 5-10 min of natural Mandarin-English conversation.
5. Stop both devices; record metadata below.

## Metadata (one row per session)
| session | date | room | speakers | duration | notes |
|---|---|---|---|---|---|
