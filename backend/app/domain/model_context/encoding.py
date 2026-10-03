"""Deterministic reversible encoding for model-visible continuity text."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

BODY_GUTTER = "  >"
_ESCAPE = "~"
_ESCAPE_PATTERN = re.compile(r"~\{([0-9A-F]{6})\}")
_SAFE_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")

_INVISIBLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF9, 0xFFFB),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE007F),
    (0xE0100, 0xE01EF),
)


def _is_invisible(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in _INVISIBLE_RANGES)


def _must_escape(character: str, *, allow_tab: bool, allow_lf: bool) -> bool:
    codepoint = ord(character)
    if character == _ESCAPE:
        return True
    if codepoint < 0x20:
        return not (
            (character == "\t" and allow_tab) or (character == "\n" and allow_lf)
        )
    return (
        0x7F <= codepoint <= 0x9F
        or codepoint in {0x2028, 0x2029}
        or 0xD800 <= codepoint <= 0xDFFF
        or _is_invisible(codepoint)
    )


def _escape_text(value: str, *, allow_tab: bool, allow_lf: bool) -> str:
    return "".join(
        f"{_ESCAPE}{{{ord(character):06X}}}"
        if _must_escape(character, allow_tab=allow_tab, allow_lf=allow_lf)
        else character
        for character in value
    )


def _unescape_text(value: str) -> str:
    result: list[str] = []
    position = 0
    while position < len(value):
        if value[position] != _ESCAPE:
            result.append(value[position])
            position += 1
            continue
        match = _ESCAPE_PATTERN.match(value, position)
        if match is None:
            raise ValueError("Malformed continuity escape sequence")
        codepoint = int(match.group(1), 16)
        if codepoint > 0x10FFFF:
            raise ValueError("Continuity escape is outside the Unicode range")
        result.append(chr(codepoint))
        position = match.end()
    return "".join(result)


def render_body_lines(value: str) -> tuple[str, ...]:
    """Render semantic record content under the fixed data gutter.

    LF is the only physical line separator. A CR from CRLF is encoded at the
    end of the preceding line, so joining and decoding restores the original
    newline convention, including mixed and trailing newlines.
    """

    encoded = _escape_text(value, allow_tab=True, allow_lf=True)
    return tuple(
        BODY_GUTTER if line == "" else f"{BODY_GUTTER} {line}"
        for line in encoded.split("\n")
    )


def decode_body_lines(lines: Sequence[str]) -> str:
    """Reverse :func:`render_body_lines` for tests and compatible clients."""

    if not lines:
        raise ValueError("At least one gutter line is required")
    encoded: list[str] = []
    for line in lines:
        if line == BODY_GUTTER:
            encoded.append("")
        elif line.startswith(f"{BODY_GUTTER} "):
            encoded.append(line[len(BODY_GUTTER) + 1 :])
        else:
            raise ValueError("Semantic body line is outside the fixed gutter")
    return _unescape_text("\n".join(encoded))


def encode_label(value: str) -> str:
    """Encode an untrusted identifier as one deterministic grammar token."""

    encoded = _escape_text(value, allow_tab=False, allow_lf=False)
    if encoded == value and _SAFE_LABEL_PATTERN.fullmatch(value):
        return value
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def decode_label(value: str) -> str:
    """Reverse :func:`encode_label`."""

    if value.startswith('"'):
        decoded = json.loads(value)
        if not isinstance(decoded, str):
            raise ValueError("Encoded continuity label must be a JSON string")
        return _unescape_text(decoded)
    if _SAFE_LABEL_PATTERN.fullmatch(value) is None:
        raise ValueError("Bare continuity label is outside the safe identifier alphabet")
    return value


def encode_prose(value: str) -> str:
    """Encode untrusted prose as one deterministic quoted grammar token."""

    encoded = _escape_text(value, allow_tab=False, allow_lf=False)
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def decode_prose(value: str) -> str:
    """Reverse :func:`encode_prose`."""

    decoded = json.loads(value)
    if not isinstance(decoded, str):
        raise TypeError("Encoded continuity prose must be a JSON string")
    return _unescape_text(decoded)
