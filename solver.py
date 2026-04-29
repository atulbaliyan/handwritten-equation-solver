import re
from dataclasses import dataclass
from typing import List

import sympy as sp


@dataclass
class SolveResult:
    mode: str
    normalized: str
    solutions: List[str]


def normalize_expr(text: str) -> str:
    t = text.strip()
    t = t.replace("×", "*").replace("÷", "/")
    t = t.replace("–", "-").replace("—", "-")
    t = t.replace(" ", "")
    t = t.replace("^", "**")
    t = t.replace("==", "=")
    t = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", t)
    return t


def solve_math(raw: str) -> SolveResult:
    expr = normalize_expr(raw)
    if not expr:
        raise ValueError("Empty expression")

    if "=" in expr:
        left, right = expr.split("=", 1)
        symbols = sorted(set(re.findall(r"[a-zA-Z]", expr)))
        if not symbols:
            val = sp.simplify(sp.sympify(left) - sp.sympify(right))
            return SolveResult("equation_check", expr, [f"difference = {val}"])

        var = sp.Symbol(symbols[0])
        eq = sp.Eq(sp.sympify(left), sp.sympify(right))
        roots = sp.solve(eq, var)
        if not roots:
            raise ValueError("No solution found")
        return SolveResult("single_equation", str(eq), [f"{var} = {sp.simplify(r)}" for r in roots])

    # Plain arithmetic
    val = sp.simplify(sp.sympify(expr))
    return SolveResult("arithmetic", expr, [f"value = {val}"])
