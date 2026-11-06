"""Evaluation.

Greedy CTC decoding: take the most likely token per frame, merge repeats, drop blanks.

Mixed error rate (MER): Chinese is scored per character, English per word, after the
same text normalization as data.py. Reports MER per test set (public test, ours).
"""

# TODO: implement
