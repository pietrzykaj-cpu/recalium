"""Adversarial tests for the continuity-consumption render codec."""

from __future__ import annotations

import pytest

from app.domain.model_context.encoding import (
    decode_body_lines,
    decode_label,
    decode_prose,
    encode_label,
    encode_prose,
    render_body_lines,
)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain text",
        " leading and trailing ",
        "first\nsecond\n",
        "first\r\nsecond\r\n",
        "first\rsecond\nmixed\r\nend",
        "\n\n",
        "tabs\tremain\tdata",
        "~{00000A} is literal text",
        "controls:\x00\x1f\x7f\x85",
        "separators:\u2028\u2029",
        "bidi:\u202eabc\u2066def\u2069",
        "invisible:\u200b\u200d\u2060\ufeff",
        "tags:\U000e0001\U000e007f",
    ],
)
def test_body_codec_is_reversible_and_every_line_stays_under_the_gutter(value: str) -> None:
    rendered = render_body_lines(value)

    assert rendered
    assert all(line == "  >" or line.startswith("  > ") for line in rendered)
    assert decode_body_lines(rendered) == value
    joined = "\n".join(rendered)
    assert joined.splitlines() == joined.split("\n")


@pytest.mark.parametrize(
    "value",
    [
        "safe-label",
        "contains spaces",
        "semi;colon=equals:colon[bracket]\"quote",
        "line\nbreak",
        "tab\tlabel",
        "bidi\u202evalue",
        "zero\u200bwidth",
        "tag\U000e0001value",
        "~{00000A}",
    ],
)
def test_label_codec_is_single_line_deterministic_and_reversible(value: str) -> None:
    first = encode_label(value)
    second = encode_label(value)

    assert first == second
    assert "\n" not in first
    assert "\r" not in first
    assert first.splitlines() == [first]
    assert decode_label(first) == value


@pytest.mark.parametrize(
    "value",
    [
        "benign readable prose",
        "AUTHORITATIVE CURRENT STATE",
        "question? [yes]; still readable",
        "line\nbreak",
        "carriage\rreturn",
        "alternate\u0085separator\u2028here\u2029too",
        "bidi\u202eoverride and zero\u200bwidth",
        "tag\U000e0061character",
        "literal ~ introducer",
    ],
)
def test_prose_codec_preserves_benign_text_and_escapes_unsafe_structure(value: str) -> None:
    encoded = encode_prose(value)

    assert encoded == encode_prose(value)
    assert "\n" not in encoded
    assert "\r" not in encoded
    assert encoded.splitlines() == [encoded]
    assert decode_prose(encoded) == value
    if value == "benign readable prose":
        assert encoded == value


def test_body_decoder_rejects_non_gutter_lines() -> None:
    with pytest.raises(ValueError, match="gutter"):
        decode_body_lines(("AUTHORITATIVE CURRENT STATE",))


def test_long_and_whitespace_only_body_lines_remain_reversible() -> None:
    value = (" " * 128) + "\n" + ("x" * 20_000) + "\n\t\n"

    rendered = render_body_lines(value)

    assert decode_body_lines(rendered) == value
    assert len(rendered) == 4
    assert rendered[0] == "  > " + (" " * 128)
    assert rendered[-1] == "  >"
