"""Turn raw multi-device recordings into utterance-level segments.

1. Align devices using the clap at the start of each session.
2. Resample every channel to 16 kHz mono.
3. Cut utterances using the corrected transcript timestamps
   (the same time spans are cut from every mic, so each utterance exists
   in lav / dr70d / esp32 versions).
"""

# TODO: implement
