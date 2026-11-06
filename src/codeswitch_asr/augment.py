"""Noise augmentation for TRAINING data only (test data is never changed).

1. Room tone: a random clip of the ESP32 room-tone recording, scaled so the
   speech-to-noise ratio equals the configured SNR (e.g. 10 dB = speech power is
   10x the noise power), then added to the clean utterance.
2. Reverberation: simulate a room with pyroomacoustics (room size, wall absorption,
   source / mic positions) and convolve the utterance with its room impulse response.

Writes the augmented training set to disk once, so training just reads it.
"""

# TODO: implement
