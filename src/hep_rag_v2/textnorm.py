from __future__ import annotations

import html
import re


TAG_RE = re.compile(r"</?[A-Za-z][^>]*?>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
SPACED_DIGIT_RE = re.compile(r"(?<=\b\d)\s+(?=\d\b)")
SPACED_LETTER_RE = re.compile(r"\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b")
OVERSCORE_RE = re.compile(r"\\(?:overline|bar|hat|tilde|vec|mathbf|boldsymbol)\s*\{\s*([^{}]+?)\s*\}")
OVERSET_RE = re.compile(r"\\(?:overset|underset|stackrel)\s*\{\s*([^{}]*?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
PHANTOM_RE = re.compile(r"\\(?:phantom|hphantom|vphantom)\s*\{\s*([^{}]*?)\s*\}")
TAU_MISSING_ARROW_RE = re.compile(r"\(\s*(τ)\s+([23])\s+(μ|e)\s*\)")

DISPLAY_LITERAL_REPAIRS = {
    "center-ofmass": "center-of-mass",
    "centerofmass": "center-of-mass",
    "protonproton": "proton-proton",
    "heavyflavor": "heavy-flavor",
    "threedimensional": "three-dimensional",
    "twodimensional": "two-dimensional",
    "datataking": "data-taking",
    "signalenriched": "signal-enriched",
}

STYLE_COMMANDS = {
    "mathrm",
    "mathit",
    "mathnormal",
    "mathbf",
    "boldsymbol",
    "bm",
    "mathsf",
    "mathtt",
    "mathbb",
    "mathcal",
    "mathscr",
    "operatorname",
    "text",
    "textrm",
    "textbf",
    "textit",
    "left",
    "right",
    "phantom",
    "hphantom",
    "vphantom",
    "dot",
    "ddot",
}

DISPLAY_COMMAND_REPLACEMENTS = {
    "to": "->",
    "rightarrow": "->",
    "leftarrow": "<-",
    "leftrightarrow": "<->",
    "pm": "±",
    "mp": "∓",
    "times": "x",
    "cdot": ".",
    "sqrt": "sqrt",
    "geq": ">=",
    "leq": "<=",
    "neq": "!=",
    "approx": "~",
    "sim": "~",
    "propto": "propto",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "eta": "η",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "ell": "ℓ",
    "cal": "",
    "mathcal": "",
}

COMMAND_REPLACEMENTS = {
    "to": "to",
    "rightarrow": "to",
    "leftarrow": "from",
    "leftrightarrow": "with",
    "pm": "plusminus",
    "mp": "minusplus",
    "times": "times",
    "cdot": "dot",
    "sqrt": "sqrt",
    "geq": "geq",
    "leq": "leq",
    "neq": "neq",
    "approx": "approx",
    "sim": "sim",
    "propto": "propto",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "eta": "eta",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "nu": "nu",
    "pi": "pi",
    "rho": "rho",
    "sigma": "sigma",
    "tau": "tau",
    "phi": "phi",
    "chi": "chi",
    "psi": "psi",
    "omega": "omega",
}

UNICODE_REPLACEMENTS = {
    "→": " to ",
    "←": " from ",
    "↔": " with ",
    "±": " plusminus ",
    "∓": " minusplus ",
    "×": " times ",
    "·": " dot ",
    "√": " sqrt ",
    "η": " eta ",
    "μ": " mu ",
    "τ": " tau ",
    "φ": " phi ",
    "π": " pi ",
    "σ": " sigma ",
    "χ": " chi ",
    "ν": " nu ",
}

DISPLAY_UNICODE_REPLACEMENTS = {
    "→": " -> ",
    "←": " <- ",
    "↔": " <-> ",
    "−": "-",
    "±": " ± ",
    "∓": " ∓ ",
    "×": " x ",
    "·": " . ",
    "√": "sqrt ",
}


def normalize_display_text(text: str) -> str:
    text = _normalize_common(text)
    text = _replace_display_unicode_math(text)
    text = _simplify_latex_wrappers(text)
    text = _replace_latex_commands(text, replacements=DISPLAY_COMMAND_REPLACEMENTS)
    text = text.replace("\\%", "%")
    text = text.replace("$", "")
    text = text.replace("^", " ^ ")
    text = text.replace("_", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.replace("\\", " ")
    text = _tighten_numeric_spacing(text)
    text = _collapse_spaced_letters(text)
    text = _tighten_exponents(text)
    text = _replace_literal_frac_sequences(text)
    text = _repair_parser_noise(text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([([])\s+([A-Za-z0-9])", r"\1\2", text)
    text = re.sub(r"([A-Za-z0-9])\s+([)\]])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_search_text(text: str) -> str:
    text = _normalize_common(text)
    text = _replace_unicode_math(text)
    text = _replace_latex_commands(text, replacements=COMMAND_REPLACEMENTS)
    text = text.replace("$", " ")
    text = text.replace("^", " ")
    text = text.replace("_", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace(":", " ")
    text = text.replace("|", " ")
    text = text.replace("-", " ")
    text = text.replace("–", " ")
    text = text.replace("/", " ")
    text = _join_spaced_digits(text)
    text = _repair_parser_noise(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_common(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\u2060", "")
    text = CONTROL_CHAR_RE.sub("", text)
    text = TAG_RE.sub(" ", text)
    return text


def _replace_unicode_math(text: str) -> str:
    for src, dst in UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def _replace_display_unicode_math(text: str) -> str:
    for src, dst in DISPLAY_UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def _replace_latex_commands(text: str, *, replacements: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        command = match.group(1)
        if command in STYLE_COMMANDS:
            return " "
        if command in replacements:
            value = replacements[command]
            return f" {value} " if value else " "
        return f" {command} "

    return LATEX_COMMAND_RE.sub(repl, text)


def _simplify_latex_wrappers(text: str) -> str:
    previous = None
    current = text
    while current != previous:
        previous = current
        current = PHANTOM_RE.sub("", current)
        current = OVERSCORE_RE.sub(r"\1", current)
        current = OVERSET_RE.sub(r"\2", current)
    return current


def _tighten_numeric_spacing(text: str) -> str:
    previous = None
    current = text
    while current != previous:
        previous = current
        current = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", current)
        current = _join_spaced_digits(current)
        current = re.sub(r"([+-])\s+(?=\d)", r"\1", current)
    return current


def _collapse_spaced_letters(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        letters = token.split()
        return "".join(letters)

    previous = None
    current = text
    while current != previous:
        previous = current
        current = SPACED_LETTER_RE.sub(repl, current)
    return current


def _tighten_exponents(text: str) -> str:
    previous = None
    current = text
    while current != previous:
        previous = current
        current = re.sub(r"([A-Za-z0-9.]+)\s*\^\s*([+-]?\d+)", r"\1^\2", current)
    return current


def _replace_literal_frac_sequences(text: str) -> str:
    if "frac" not in text:
        return text
    tokens = text.split()
    if not tokens:
        return text

    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "frac":
            out.append(tokens[idx])
            idx += 1
            continue
        rendered, next_idx = _parse_literal_frac(tokens, idx)
        out.extend(rendered)
        idx = next_idx
    return " ".join(out)


def _parse_literal_frac(tokens: list[str], idx: int, *, depth: int = 0) -> tuple[list[str], int]:
    if idx >= len(tokens) or tokens[idx] != "frac" or depth > 4:
        return (["frac"], min(idx + 1, len(tokens)))

    numerator, next_idx = _parse_literal_frac_operand(tokens, idx + 1, depth=depth + 1)
    denominator, end_idx = _parse_literal_frac_operand(tokens, next_idx, depth=depth + 1)
    if not numerator or not denominator:
        return (["frac"], min(idx + 1, len(tokens)))
    return (["("] + numerator + [")", "/", "("] + denominator + [")"], end_idx)


def _parse_literal_frac_operand(tokens: list[str], idx: int, *, depth: int) -> tuple[list[str], int]:
    if idx >= len(tokens):
        return [], idx
    token = tokens[idx]
    if token == "frac":
        return _parse_literal_frac(tokens, idx, depth=depth)
    if token in {".", ",", ";", ":", "!", "?", ")", "]"}:
        return [], idx

    out = [token]
    idx += 1
    while idx < len(tokens):
        token = tokens[idx]
        if re.fullmatch(r"\([^()]*\)|\[[^\[\]]*\]", token):
            out.append(token)
            idx += 1
            continue
        if token in {"^", "_"} and idx + 1 < len(tokens):
            out.extend([token, tokens[idx + 1]])
            idx += 2
            continue
        if token in {"+", "-", "±", "∓"} and idx + 1 < len(tokens):
            out.extend([token, tokens[idx + 1]])
            idx += 2
            continue
        break
    return out, idx


def _join_spaced_digits(text: str) -> str:
    previous = None
    current = text
    while current != previous:
        previous = current
        current = SPACED_DIGIT_RE.sub("", current)
    return current


def _repair_parser_noise(text: str) -> str:
    for src, dst in DISPLAY_LITERAL_REPAIRS.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    text = TAU_MISSING_ARROW_RE.sub(r"(τ -> \2 \3)", text)
    text = re.sub(r"\b([A-Za-z]{2,})-\s*>\.?\s*([A-Za-z]{2,})\b", r"\1\2", text)
    text = re.sub(r"\s*\^\s*\.,(?=\s)", ",", text)
    text = re.sub(r"\s*\^\s*,(?=\s)", ",", text)
    text = re.sub(r"\s*\^\s*\.(?=\s|$)", "", text)
    text = re.sub(r"\bpT\s*,\s*,(?=\s|$)", "pT,", text)
    text = re.sub(r"\)\s*\.\s*,(?=\s)", "),", text)
    text = re.sub(r"(?<=\d)\s*\.\s*,(?=\s)", ",", text)
    text = re.sub(r"(?<=\d)\s*,\s*,(?=\s)", ",", text)
    text = re.sub(r"(?<=\d)\s*,\s*\.(?=\s|$)", ".", text)
    text = re.sub(r"(?<=[A-Z])\s*,\s*,(?=\s)", ",", text)
    text = re.sub(r"(?<=[A-Z])\s*,\s*\.(?=\s|$)", ".", text)
    text = re.sub(r"(?<=p)\s*,\s*,(?=\s)", ",", text)
    text = re.sub(r"(?<=\S)\s*,\s*,(?=\s)", ",", text)
    text = re.sub(r"(?<=\w),{2,}", ",", text)
    text = re.sub(r"(?<=\w)\.{2,}", ".", text)
    text = re.sub(r"\bbegin\s+array\b", " ", text)
    text = re.sub(r"\bend\s+array\b", " ", text)
    text = re.sub(r"([a-z])sqrt\b", r"\1 sqrt", text)
    text = re.sub(r"\bsqrt\s+(?:rmT|T)\s*=\s*sqrt\b", "mT = sqrt", text)
    text = re.sub(r"\bmass\s+sqrt\s+(?:rmT|T)\s*=\s*sqrt\b", "mass mT = sqrt", text)
    text = re.sub(
        r"\btransverse\s+mass(?:\s+is)?\s+(?:rmT|T)\s*=\s*sqrt\b",
        "transverse mass is mT = sqrt",
        text,
    )
    text = re.sub(r"\brm([A-Z])\b", r"\1", text)
    text = re.sub(r"\bpp(?=B\b|Ds\b|D\b|W\b)", "pp ", text)
    text = re.sub(r"\bB(?=Ds\b)", "B ", text)
    text = re.sub(r"\ba(?=Ds\b|W\b)", "a ", text)
    text = re.sub(r"\bpp\s+W\s+\+\s+X\b", "pp -> W + X", text)
    text = re.sub(r"\bpp\s+B\s+\+\s+X\b", "pp -> B + X", text)
    text = re.sub(r"\bpp\s+Ds\s*\^\s*\+\s*\+\s+X\b", "pp -> Ds ^ + + X", text)
    text = re.sub(r"\bB\s+τ\s+\+\s+X\b", "B -> τ + X", text)
    text = re.sub(r"\bB\s+Ds\s*\^\s*\+\s*\+\s+X\b", "B -> Ds ^ + + X", text)
    text = re.sub(r"\bW\s+τ\s+ν\s+τ\b", "W -> τ ν τ", text)
    text = re.sub(r"\bLm(?=\s*\()", "L m", text)
    text = re.sub(r"\bJ\s*/\s*ψ\b", "J/ψ", text)
    text = re.sub(r"\bab hadron\b", "a b hadron", text)
    text = re.sub(
        r",\s*detailed\s+in\s+(?:Ref\.\s*)?,?\s*is\s+summarized\s+here\b",
        " is summarized here",
        text,
    )
    text = re.sub(r"\bdetailed\s+in\.(?=\s|$)", "", text)
    text = re.sub(r"\b(?:can\s+be\s+)?found\s+in\.(?=\s|$)", "", text)
    text = re.sub(r"\bdetailed\s+in,\s+is\s+summarized\s+here\b", "is summarized here", text)
    text = re.sub(r"\bdetailed\s+in\s+is\s+summarized\s+here\b", "is summarized here", text)
    text = re.sub(r",\s+is\s+summarized\s+here\b", " is summarized here", text)
    text = re.sub(r"\bDelta\s+R\s*:\s*<\s*:\s*", "Delta R < ", text)
    text = re.sub(r"\btrigB\s*\(", "trig B (", text)
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "-", text)
    text = re.sub(r"(?<=\d\.\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=\d)\.(?=\d[A-Za-z])", ". ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"(?<=\S)\s*,\s*,(?=\s|$)", ",", text)
    text = re.sub(r"\.\s*\.(?=\s|$)", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text
