"""
AST-based calculator tool for safe arithmetic evaluation.
Gemini 3 handles basic arithmetic well natively; this tool adds reliability for multi-step
financial calculations and uses templated examples in the description to guide effective tool use.
"""

import ast
import operator as op

from langchain.tools import tool
from langgraph.prebuilt import ToolNode

# --- Calculator Tool ---

# Whitelisted operations; anything outside this raises ValueError in _safe_eval
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node using only the whitelisted _OPS operators."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool()
def calculate(expression: str) -> str:
    """
    Evaluates arithmetic expressions to solve financial reasoning tasks.
    Use this tool to derive metrics from values extracted from text and tables.

    Conceptual templates for financial conventions:
    - Portion/Contribution: Segment_Value / Total_Value (to find share of total)
    - Year-over-Year Change: (Current_Year - Prior_Year) / Prior_Year
    - Margin Analysis: (Total_Revenue - Cost_of_Goods_Sold) / Total_Revenue
    - Scaling/Normalization: Value_in_Millions / 1000 (to convert to Billions)
    - Net Position: Positive_Inflow + (Negative_Outflow)
    - Average/Weighted Portion: (Value_A + Value_B) / Number_of_Periods

    Note: Convert values in parentheses (e.g., '(500)') to negative numbers (e.g., '-500')
    before passing to the expression.

    Args:
        expression: A string math expression (e.g., 'segment_value / total_value').
    """
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


tools = [calculate]
tool_node = ToolNode(tools, handle_tool_errors=True)
