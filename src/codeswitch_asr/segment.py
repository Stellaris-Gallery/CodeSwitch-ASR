"""Turn raw session recordings into utterance-level segments.

1. Align the ESP32 and phone recordings using the clap at the start of each session.
2. Resample to 16 kHz mono.
3. Cut utterances using the corrected transcript timestamps (the phone recording is
   only used to hear the speech clearly while transcribing; the ESP32 audio is what
   goes into the experiments).
"""

# TODO: implement
