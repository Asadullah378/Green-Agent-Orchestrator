"""
Green Agent Orchestrator (GAO) — Agent tools

Deterministic tools with hardcoded data so experiments are fully reproducible
without external API calls.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timedelta

from langchain_core.tools import tool

# ── Safe math evaluator ─────────────────────────────────────────────────────

_ALLOWED_NAMES: dict = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": pow, "log": math.log, "log10": math.log10,
    "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
}

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only allowed operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are allowed")
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Name):
        val = _ALLOWED_NAMES.get(node.id)
        if val is None:
            raise ValueError(f"Name not allowed: {node.id}")
        return val
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, //, %, ** and functions sqrt(), pow(), log(),
    log10(), abs(), round(), min(), max(), ceil(), floor(). Constants: pi, e.
    Examples: '15 * 230 / 100', 'sqrt(144)', 'round(3.14159, 2)'.
    """
    try:
        expr = expression.strip().replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ── Unit converter ───────────────────────────────────────────────────────────

_CONVERSION_RATES: dict[str, dict[str, float]] = {
    # Currency (rates fixed for reproducibility — notional mid-market)
    "USD": {"EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "USD": 1.0},
    "EUR": {"USD": 1.087, "GBP": 0.859, "JPY": 162.5, "EUR": 1.0},
    "GBP": {"USD": 1.266, "EUR": 1.164, "JPY": 189.1, "GBP": 1.0},
    "JPY": {"USD": 0.00669, "EUR": 0.00615, "GBP": 0.00529, "JPY": 1.0},
    # Distance
    "km":   {"miles": 0.621371, "m": 1000, "ft": 3280.84, "km": 1.0},
    "miles": {"km": 1.60934, "m": 1609.34, "ft": 5280, "miles": 1.0},
    "m":    {"km": 0.001, "miles": 0.000621371, "ft": 3.28084, "m": 1.0},
    "ft":   {"m": 0.3048, "km": 0.0003048, "miles": 0.000189394, "ft": 1.0},
    # Weight
    "kg":  {"lbs": 2.20462, "g": 1000, "oz": 35.274, "kg": 1.0},
    "lbs": {"kg": 0.453592, "g": 453.592, "oz": 16, "lbs": 1.0},
    "g":   {"kg": 0.001, "lbs": 0.00220462, "oz": 0.035274, "g": 1.0},
    "oz":  {"kg": 0.0283495, "lbs": 0.0625, "g": 28.3495, "oz": 1.0},
}


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between units.

    Supported unit families:
      Currency : USD, EUR, GBP, JPY
      Distance : km, miles, m, ft
      Weight   : kg, lbs, g, oz
    Temperature is handled separately — pass from_unit/to_unit as 'C' or 'F'.
    """
    fu, tu = from_unit.strip(), to_unit.strip()

    # Temperature special case
    if fu.upper() in ("C", "CELSIUS") and tu.upper() in ("F", "FAHRENHEIT"):
        return str(round(value * 9 / 5 + 32, 2))
    if fu.upper() in ("F", "FAHRENHEIT") and tu.upper() in ("C", "CELSIUS"):
        return str(round((value - 32) * 5 / 9, 2))

    rates = _CONVERSION_RATES.get(fu)
    if rates is None:
        return f"Error: unknown source unit '{fu}'"
    factor = rates.get(tu)
    if factor is None:
        return f"Error: cannot convert from '{fu}' to '{tu}'"
    result = round(value * factor, 4)
    return f"{result} {tu}"


# ── Mock database ────────────────────────────────────────────────────────────

MOCK_DB: dict[str, list[dict]] = {
    "products": [
        {"id": 1,  "name": "Wireless Mouse",      "category": "electronics", "price": 29.99,  "stock": 150},
        {"id": 2,  "name": "USB-C Hub",            "category": "electronics", "price": 49.99,  "stock": 75},
        {"id": 3,  "name": "Mechanical Keyboard",  "category": "electronics", "price": 89.99,  "stock": 40},
        {"id": 4,  "name": "Laptop Stand",         "category": "accessories", "price": 34.99,  "stock": 120},
        {"id": 5,  "name": "Webcam HD",            "category": "electronics", "price": 59.99,  "stock": 60},
        {"id": 6,  "name": "Desk Lamp",            "category": "accessories", "price": 24.99,  "stock": 200},
        {"id": 7,  "name": "Monitor 27″",          "category": "electronics", "price": 299.99, "stock": 25},
        {"id": 8,  "name": "Notebook Pack",         "category": "stationery",  "price": 12.99,  "stock": 500},
        {"id": 9,  "name": "Ergonomic Chair",      "category": "furniture",   "price": 249.99, "stock": 15},
        {"id": 10, "name": "Cable Organizer",       "category": "accessories", "price": 9.99,   "stock": 300},
    ],
    "companies": [
        {"name": "ACME Corp",   "sector": "Technology", "employees": 1200, "q1_revenue": 1100000, "q2_revenue": 1300000, "q3_revenue": 1500000, "q4_revenue": 1800000},
        {"name": "Globex Inc",  "sector": "Finance",    "employees": 800,  "q1_revenue": 900000,  "q2_revenue": 950000,  "q3_revenue": 870000,  "q4_revenue": 1020000},
        {"name": "Initech",     "sector": "Consulting", "employees": 350,  "q1_revenue": 500000,  "q2_revenue": 520000,  "q3_revenue": 540000,  "q4_revenue": 560000},
        {"name": "Umbrella Ltd","sector": "Healthcare", "employees": 2500, "q1_revenue": 3200000, "q2_revenue": 3100000, "q3_revenue": 3400000, "q4_revenue": 3600000},
        {"name": "Stark Ind",   "sector": "Manufacturing","employees": 5000,"q1_revenue": 8000000,"q2_revenue": 8500000,"q3_revenue": 9000000,"q4_revenue": 9500000},
    ],
    "employees": [
        {"id": 1,  "name": "Alice Johnson",  "department": "Engineering", "salary": 95000,  "hire_date": "2021-03-15"},
        {"id": 2,  "name": "Bob Smith",      "department": "Marketing",   "salary": 72000,  "hire_date": "2020-07-01"},
        {"id": 3,  "name": "Carol Williams", "department": "Engineering", "salary": 105000, "hire_date": "2019-11-20"},
        {"id": 4,  "name": "David Brown",    "department": "Sales",       "salary": 68000,  "hire_date": "2022-01-10"},
        {"id": 5,  "name": "Eve Davis",      "department": "Engineering", "salary": 115000, "hire_date": "2018-05-22"},
        {"id": 6,  "name": "Frank Miller",   "department": "HR",          "salary": 65000,  "hire_date": "2023-02-14"},
        {"id": 7,  "name": "Grace Wilson",   "department": "Marketing",   "salary": 78000,  "hire_date": "2021-09-01"},
        {"id": 8,  "name": "Henry Taylor",   "department": "Sales",       "salary": 71000,  "hire_date": "2020-04-18"},
        {"id": 9,  "name": "Ivy Anderson",   "department": "Engineering", "salary": 98000,  "hire_date": "2022-06-30"},
        {"id": 10, "name": "Jack Thomas",    "department": "Finance",     "salary": 88000,  "hire_date": "2019-08-12"},
    ],
}


@tool
def data_lookup(table: str, column: str = "", value: str = "") -> str:
    """Look up rows from a mock database.

    Available tables: products, companies, employees.
    Optionally filter by column == value.
    If no filter is given, returns all rows. Returns JSON-formatted results.
    """
    import json

    tbl = MOCK_DB.get(table.strip().lower())
    if tbl is None:
        return f"Error: unknown table '{table}'. Available: {list(MOCK_DB.keys())}"

    if column and value:
        col = column.strip().lower()
        val = value.strip()
        rows = []
        for row in tbl:
            cell = str(row.get(col, ""))
            if cell.lower() == val.lower():
                rows.append(row)
        if not rows:
            return f"No rows found where {col} == '{val}' in table '{table}'."
        return json.dumps(rows, indent=2)

    return json.dumps(tbl, indent=2)


# ── Date calculator ──────────────────────────────────────────────────────────

_DATE_FMT = "%Y-%m-%d"


@tool
def date_calculator(operation: str, date1: str, date2: str = "", days: int = 0) -> str:
    """Perform date arithmetic.

    Operations:
      'diff'     — days between date1 and date2 (format YYYY-MM-DD)
      'add'      — add *days* to date1
      'subtract' — subtract *days* from date1
      'weekday'  — return the weekday name of date1
    """
    try:
        d1 = datetime.strptime(date1.strip(), _DATE_FMT)
    except ValueError:
        return f"Error: cannot parse date1 '{date1}'. Use YYYY-MM-DD."

    op = operation.strip().lower()

    if op == "diff":
        try:
            d2 = datetime.strptime(date2.strip(), _DATE_FMT)
        except ValueError:
            return f"Error: cannot parse date2 '{date2}'. Use YYYY-MM-DD."
        delta = abs((d2 - d1).days)
        return f"{delta} days"

    if op == "add":
        result = d1 + timedelta(days=days)
        return result.strftime(_DATE_FMT)

    if op == "subtract":
        result = d1 - timedelta(days=days)
        return result.strftime(_DATE_FMT)

    if op == "weekday":
        return d1.strftime("%A")

    return f"Error: unknown operation '{op}'. Use diff, add, subtract, or weekday."


# ── Text processor ───────────────────────────────────────────────────────────

@tool
def text_processor(operation: str, text: str) -> str:
    """Process text with simple string operations.

    Operations: uppercase, lowercase, word_count, char_count,
    reverse, extract_numbers, title_case, strip.
    """
    import re

    op = operation.strip().lower()
    t = text

    if op == "uppercase":
        return t.upper()
    if op == "lowercase":
        return t.lower()
    if op == "word_count":
        return str(len(t.split()))
    if op == "char_count":
        return str(len(t))
    if op == "reverse":
        return t[::-1]
    if op == "extract_numbers":
        nums = re.findall(r"-?\d+\.?\d*", t)
        return ", ".join(nums) if nums else "No numbers found."
    if op == "title_case":
        return t.title()
    if op == "strip":
        return t.strip()

    return f"Error: unknown operation '{op}'."


# ── Convenience list of all tools ────────────────────────────────────────────

ALL_TOOLS = [calculator, unit_converter, data_lookup, date_calculator, text_processor]
