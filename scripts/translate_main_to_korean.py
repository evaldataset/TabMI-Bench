from __future__ import annotations

import re
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "main.tex"
DST = ROOT / "paper" / "korean.tex"

TRANS_CMDS = {"title", "section", "subsection", "subsubsection", "paragraph", "subparagraph", "caption", "textbf", "textit", "emph"}
KEEP_BLOCK_PREFIXES = (
    r"\begin{",
    r"\end{",
    r"\label{",
    r"\includegraphics",
    r"\centering",
    r"\small",
    r"\scriptsize",
    r"\footnotesize",
    r"\toprule",
    r"\midrule",
    r"\bottomrule",
    r"\appendix",
    r"\newpage",
    r"\maketitle",
)
PLACEHOLDER_RE = re.compile(r"ZXPH(\d+)ZX")
MATH_RE = re.compile(r"(\$[^$]*\$|\\\([^\)]*\\\)|\\\[[^\]]*\\\])", re.DOTALL)
PROTECT_CMD_RE = re.compile(
    r"(\\(?:cite|citep|citet|citeauthor|ref|eqref|label|url|texttt|checkmark|triangle|times|answerYes|answerNo|answerNA)\*?(?:\[[^\]]*\])?\{[^{}]*\})"
)

cache: dict[str, str] = {}


def translate_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return text
    if normalized in cache:
        return cache[normalized]
    response = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": normalized},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    translated = "".join(chunk[0] for chunk in payload[0])
    cache[normalized] = translated
    return translated


def find_matching_brace(text: str, start: int) -> int:
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError("Unmatched brace")


def translate_command_args(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue
        cmd_match = re.match(r"\\([A-Za-z@]+)\*?(?:\[[^\]]*\])?\{", text[i:])
        if not cmd_match:
            out.append(text[i])
            i += 1
            continue
        cmd = cmd_match.group(1)
        if cmd not in TRANS_CMDS:
            out.append(text[i : i + cmd_match.end()])
            i += cmd_match.end()
            continue
        head = text[i : i + cmd_match.end()]
        brace_start = i + cmd_match.end() - 1
        brace_end = find_matching_brace(text, brace_start)
        arg = text[brace_start + 1 : brace_end]
        out.append(head + translate_inline(arg) + "}")
        i = brace_end + 1
    return "".join(out)


def protect(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def repl(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"ZXPH{len(tokens)-1}ZX"

    text = MATH_RE.sub(repl, text)
    text = PROTECT_CMD_RE.sub(repl, text)
    text = re.sub(r"(\\(?:begin|end)\{[^{}]*\})", repl, text)
    text = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\})", repl, text)
    return text, tokens


def restore(text: str, tokens: list[str]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: tokens[int(m.group(1))], text)


def escape_plain(text: str) -> str:
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    return text


def translate_inline(text: str) -> str:
    if not re.search(r"[A-Za-z]", text):
        return text
    masked, tokens = protect(text)
    if not re.search(r"[A-Za-z]", masked):
        return restore(masked, tokens)
    translated = translate_text(masked)
    translated = escape_plain(translated)
    return restore(translated, tokens)


def translate_table_line(line: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    tail = ""
    if body.rstrip().endswith(r"\\"):
        stripped = body.rstrip()
        tail = r"\\"
        body = stripped[:-2]
    parts = body.split("&")
    parts = [translate_inline(part) for part in parts]
    return " & ".join(parts) + tail + newline


def translate_block(block: str) -> str:
    stripped = block.strip()
    if not stripped:
        return block
    if stripped.startswith("%"):
        return block
    if stripped.startswith(KEEP_BLOCK_PREFIXES):
        return block
    if "&" in block:
        return "".join(translate_table_line(line) for line in block.splitlines(keepends=True))
    if stripped.startswith(r"\item"):
        prefix = block[: block.index(r"\item") + len(r"\item")]
        rest = block[block.index(r"\item") + len(r"\item") :]
        return prefix + translate_inline(translate_command_args(rest))
    if stripped.startswith("\\"):
        return translate_command_args(block)
    return translate_inline(translate_command_args(block))


def split_body(body: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", body)
    blocks: list[str] = []
    current = ""
    for part in parts:
        if re.fullmatch(r"\n\s*\n", part):
            if current:
                blocks.append(current)
                current = ""
            blocks.append(part)
        else:
            current += part
    if current:
        blocks.append(current)
    return blocks


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    preamble, body = text.split(r"\begin{document}", 1)
    preamble = preamble.replace(
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[utf8]{inputenc}" + "\n" + r"\usepackage{kotex}",
        1,
    )
    preamble = translate_command_args(preamble)

    blocks = split_body(body)
    translated_blocks = [translate_block(block) for block in blocks]
    output = preamble + r"\begin{document}" + "".join(translated_blocks)
    DST.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
