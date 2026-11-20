"""Tests for the digit-banning logic (no model download needed)."""

from codeswitch_asr.transcribe import find_digit_token_ids, has_digit


def test_has_digit():
    assert has_digit("我买了711的咖啡")
    assert has_digit("２０２５")            # full-width digits count too
    assert not has_digit("七百一十一")       # Chinese numerals are words, not digits
    assert not has_digit("SEVEN ELEVEN")


class FakeTokenizer:
    """Tiny stand-in for WhisperTokenizer: 6 ordinary tokens + 2 special ones."""
    pieces = ["我", "711", " x2", "hello", "<|zh|>", "<|0.00|>", "十一", "2025"]
    all_special_ids = [4]          # <|zh|>; <|0.00|> is caught by the "<|...|>" check

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, i):
        return self.pieces[i]

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_find_digit_token_ids_skips_special_tokens():
    # "711", " x2", "2025" are banned; the timestamp token <|0.00|> is NOT
    assert find_digit_token_ids(FakeTokenizer()) == [1, 2, 7]
