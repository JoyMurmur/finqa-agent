"""Tests for the AST-based calculator tool.

Custom logic with no external dependencies. Covers the financial patterns from
the tool docstring and the security edge case (no eval/exec).
"""

import pytest

from src.agent.tools import calculate


def _calc(expr: str) -> float:
    return float(calculate.invoke({"expression": expr}))


def test_basic_arithmetic():
    assert _calc("(10 - 3) * 2 / 4") == pytest.approx(3.5)


def test_yoy_change():
    # (current - prior) / prior — most common pattern in ConvFinQA
    assert _calc("(18.1 - 14.6) / 14.6") == pytest.approx(0.23972602739726, rel=1e-5)


def test_margin_with_negative():
    # losses shown as negatives, e.g. (500) -> -500
    assert _calc("(100 + -40) / 100") == pytest.approx(0.6)


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        _calc("1 / 0")


def test_unsafe_expression_raises():
    with pytest.raises(Exception):
        _calc("__import__('os').system('ls')")
