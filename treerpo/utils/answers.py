
import regex as re
from fractions import Fraction
from math import isclose
from typing import Union, Optional


try:
    import sympy
    from sympy import N, simplify
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    _SYM_AVAILABLE = True
except Exception:
    sympy = None  # type: ignore
    parse_expr = None  # type: ignore
    parse_latex = None  # type: ignore
    _SYM_AVAILABLE = False

try:
    from latex2sympy2 import latex2sympy
    _LATEX2_AVAILABLE = True
except Exception:
    latex2sympy = None  # type: ignore
    _LATEX2_AVAILABLE = False

__all__ = ["extract_final_answer", "math_equal"]

# ---------------- extract_final_answer ----------------
_BOX = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL | re.IGNORECASE)

def extract_final_answer(llm_output: str) -> Optional[str]:
    m = _BOX.search(llm_output or "")
    if m:
        ans = m.group(1).strip()
        # strip accidental \begin...\end...
        ans = re.sub(r"\\begin\{.*?\}(.*?)\\end\{.*?\}", r"\1", ans, flags=re.DOTALL).strip()
        return ans
    # fallback: last line
    lines = (llm_output or "").strip().split("\n")
    return lines[-1].strip() if lines else None

# ---------------- math_equal + helpers ----------------
_MC_LETTERS = {"A", "B", "C", "D", "E"}
_COMMA = re.compile(",")
_PCT = re.compile(r"\\?%$")  # '%' or '\%'
_TEXT_BLOCK = re.compile(r"\\text\{.*?\}", re.DOTALL)

def _choice_answer_clean(pred: str) -> str:
    pred = (pred or "").strip().strip(".:/").strip()
    letters = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    return (letters[-1] if letters else pred.strip().strip("."))

def _parse_digits(s: str) -> Optional[float]:
    s = _COMMA.sub("", s.strip())
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", s):
        try:
            return float(Fraction(s.replace(" ", "")))
        except Exception:
            pass
    pct = False
    if _PCT.search(s):
        s = _PCT.sub("", s)
        pct = True
    try:
        v = float(s)
        return v / 100.0 if pct else v
    except ValueError:
        return None

def _numeric_equal(a: float, b: float) -> bool:
    return isclose(a, b, rel_tol=1e-4)

def _clean_latex(s: str) -> str:
    s = s.replace("\\\\", "\\").replace("$", "").replace("\\$", "")
    s = _TEXT_BLOCK.sub("", s)
    return s.strip()

def _sympy_parse(s: str):
    if not _SYM_AVAILABLE:
        return s
    s = _clean_latex(str(s))
    parsers = []
    if "\\" in s and parse_latex is not None:
        parsers.append(parse_latex)
    if _LATEX2_AVAILABLE and latex2sympy is not None:
        parsers.append(latex2sympy)
    parsers.append(lambda x: parse_expr(x, evaluate=False))
    for f in parsers:
        try:
            obj = f(s)
            if hasattr(sympy, "Expr") and isinstance(obj, sympy.Expr):
                obj = obj.subs({sympy.Symbol("pi"): sympy.pi, sympy.Symbol("i"): sympy.I})
            return obj
        except Exception:
            continue
    return s

def _symbolic_equal(a: str, b: str) -> bool:
    if not _SYM_AVAILABLE:
        return False
    A, B = _sympy_parse(a), _sympy_parse(b)
    if isinstance(A, str) or isinstance(B, str):
        return str(A) == str(B)
    try:
        if A == B:
            return True
    except Exception:
        pass
    try:
        if simplify(A - B).equals(0):
            return True
    except Exception:
        pass
    try:
        if hasattr(A, "equals") and A.equals(B):
            return True
    except Exception:
        pass
    try:
        if isinstance(A, sympy.Eq) and isinstance(B, sympy.Eq):
            dA, dB = A.lhs - A.rhs, B.lhs - B.rhs
            if simplify(dA - dB).equals(0) or simplify(dA + dB).equals(0):
                return True
    except Exception:
        pass
    try:
        aN, bN = float(N(A)), float(N(B))
        if _numeric_equal(aN, bN):
            return True
    except Exception:
        pass
    try:
        if isinstance(A, sympy.Matrix) and isinstance(B, sympy.Matrix):
            if A.shape == B.shape and simplify(A - B).equals(sympy.zeros(*A.shape)):
                return True
    except Exception:
        pass
    return False

def _str_to_pmatrix(s: str) -> str:
    s = s.strip().strip("()[]")
    rows = re.findall(r"\{.*?\}|\[.*?\]", s)
    if not rows:
        return s
    out_rows = []
    for r in rows:
        r = r.strip("{}[]")
        elems = re.split(r",\s*|\s+", r.strip())
        out_rows.append(" & ".join(elems))
    return r"\begin{pmatrix}" + r" \\ ".join(out_rows) + r"\end{pmatrix}"

def math_equal(
    prediction: Union[bool, float, int, str, None],
    reference: Union[bool, float, int, str, None],
    include_percentage: bool = True,
    is_close_numerical: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False
    pred = str(prediction).strip()
    ref  = str(reference).strip()
    if pred.lower() == ref.lower():
        return True
    if ref in _MC_LETTERS:
        return _choice_answer_clean(pred) == ref
    p_num, r_num = _parse_digits(pred), _parse_digits(ref)
    if p_num is not None and r_num is not None:
        candidates = {r_num}
        if include_percentage:
            candidates |= {r_num / 100.0, r_num * 100.0}
        for c in candidates:
            if is_close_numerical:
                if _numeric_equal(p_num, c): return True
            else:
                if p_num == c: return True
    if pred.count("=") == 1 and ref.count("=") == 0:
        try:
            _, pv = [t.strip() for t in pred.split("=", 1)]
            if math_equal(pv, ref, include_percentage, is_close_numerical): return True
        except Exception: pass
    if ref.count("=") == 1 and pred.count("=") == 0:
        try:
            _, rv = [t.strip() for t in ref.split("=", 1)]
            if math_equal(pred, rv, include_percentage, is_close_numerical): return True
        except Exception: pass
    if (pred.startswith("(") and pred.endswith(")") and ref.startswith("[") and ref.endswith("]")) or \
       (pred.startswith("[") and pred.endswith("]") and ref.startswith("(") and ref.endswith(")")):
        sp, sr = pred[1:-1].strip(), ref[1:-1].strip()
        if sp == sr or _symbolic_equal(sp, sr): return True
    is_pred_mat = (r'\begin{pmatrix}' in pred) or (r'\begin{bmatrix}' in pred)
    is_ref_mat  = (r'\begin{pmatrix}' in ref)  or (r'\begin{bmatrix}' in ref)
    if is_pred_mat and not is_ref_mat and "{" in ref:
        ref = _str_to_pmatrix(ref)
    elif is_ref_mat and not is_pred_mat and "{" in pred:
        pred = _str_to_pmatrix(pred)
    return _symbolic_equal(pred, ref)
