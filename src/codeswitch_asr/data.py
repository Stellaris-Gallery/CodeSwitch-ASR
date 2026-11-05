"""Datasets in one common format.

- ASCEND (Hugging Face: CAiRE/ASCEND), public, CC BY-SA 4.0.
- Our own recordings, after segmentation (see segment.py).

Every utterance becomes a record like:
    {"audio": array (16 kHz), "text": str, "conversation_id": str,
     "speaker_id": str, "mic": "lav" | "dr70d" | "esp32"}

Train/test splits of our recordings are by WHOLE conversation, never by utterance.
"""

# TODO: implement
