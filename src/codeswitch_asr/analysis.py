"""Error analysis.

1. Switch-point error rate: find Mandarin/English switch points in the reference text,
   align prediction and reference (jiwer), and compare the error rate within a few
   tokens of a switch vs. everywhere else.
2. CTC frame-probability plot: per-frame token probabilities for an utterance
   (time x token heatmap), especially around switch points.
"""

# TODO: implement
