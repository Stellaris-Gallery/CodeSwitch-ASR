"""Tests for text normalization. Each expected value was worked out by hand."""

from codeswitch_asr.normalize import has_unintelligible, normalize, tokenize


def test_example_from_notes():
    # Same sentence, two writing styles -> identical after normalization.
    ref = "这个 PROJECT 的 DEADLINE 是明天<FIL/>。"     # CS-Dialogue style
    hyp = "這個project的deadline是明天。"               # Whisper style
    expected = "这 个 PROJECT 的 DEADLINE 是 明 天"
    assert normalize(ref) == expected
    assert normalize(hyp) == expected


def test_all_four_tags_removed():
    assert normalize("<FIL/>好<SPK/>的<NON/>呀<NPS/>") == "好 的 呀"


def test_unintelligible_marker():
    assert normalize("我觉得**挺好的") == "我 觉 得 挺 好 的"
    assert has_unintelligible("我觉得**挺好的")
    assert not has_unintelligible("我觉得挺好的")


def test_punctuation_chinese_and_english():
    assert normalize("啊，好，嗨，贝拉，嗯。") == "啊 好 嗨 贝 拉 嗯"
    assert normalize("Yeah, yeah... OK?!") == "YEAH YEAH OK"


def test_apostrophe_kept_inside_word_only():
    assert normalize("I don't know") == "I DON'T KNOW"
    assert normalize("I don’t know") == "I DON'T KNOW"     # curly apostrophe
    assert normalize("'cool'") == "COOL"


def test_acronym_letters_merged():
    # Reference spells acronyms with spaces; Whisper writes them together.
    assert normalize("我用 I B M 的 LAPTOP") == "我 用 IBM 的 LAPTOP"
    assert normalize("我用IBM的laptop") == "我 用 IBM 的 LAPTOP"


def test_single_letter_word_not_merged():
    # One letter alone (no neighbour letter) stays as it is.
    assert normalize("I think 是 A 啦") == "I THINK 是 A 啦"


def test_digits_left_alone():
    # Digits are handled when Whisper generates text, not here.
    assert normalize("我住711") == "我 住 711"


def test_full_width_characters():
    assert normalize("ＯＫ，好的") == "OK 好 的"


def test_tokenize_returns_list():
    assert tokenize("Hello世界") == ["HELLO", "世", "界"]


def test_empty_after_normalization():
    assert normalize("<FIL/>。") == ""
