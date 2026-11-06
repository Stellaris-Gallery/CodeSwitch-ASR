"""CTC fine-tuning entry point.

Usage:
    uv run python -m codeswitch_asr.finetune --config configs/ft_ascend.yaml

Loads a pretrained speech encoder (wav2vec 2.0 family) with a new linear CTC output
layer sized to our vocabulary (Hugging Face Wav2Vec2ForCTC; the CTC loss is the
existing implementation), fine-tunes it on the datasets listed in the config, and
logs loss curves, per-epoch MER, and the full config to Weights & Biases.
"""

# TODO: implement
