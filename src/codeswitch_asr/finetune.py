"""Fine-tuning entry point.

Usage:
    uv run python -m codeswitch_asr.finetune --config configs/ft_ascend.yaml

Fine-tunes a pretrained ASR model (full or LoRA via peft) on the datasets
listed in the config. Logs loss curves, per-epoch MER, and the full config
to Weights & Biases.
"""

# TODO: implement
