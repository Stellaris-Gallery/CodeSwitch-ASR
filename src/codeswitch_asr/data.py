"""Datasets in one common format.

- Public corpus (TBD; ASCEND is the current candidate -- Hugging Face: CAiRE/ASCEND).
- Our own ESP32 recordings, after segmentation (see segment.py).

Every utterance becomes a record like:
    {"audio": array (16 kHz), "text": str, "conversation_id": str, "speaker_id": str}

Also here:
- Text normalization lives in normalize.py (shared with MER). Use it for BOTH the
  CTC vocabulary and MER. Drop TRAINING utterances where
  normalize.has_unintelligible(text) is True (transcript has "**").
- CTC vocabulary: Chinese characters + English letters + word delimiter "|" + blank.

Train/test splits of our recordings are by WHOLE conversation, never by utterance.
"""

# TODO: implement
