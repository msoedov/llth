import ast
import importlib
import io
import json
import pkgutil
import subprocess
import sys
import time
import tokenize
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from openai import OpenAI

import lessons
from evals.capture import capture
from evals.checks import Check

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
JUDGE_PROMPT = (
    "You compare two answers to the same input. They agree unless one contradicts the other or "
    "leaves out a fact the user needs. Ignore wording, tone, formatting, and extra detail."
)
VERDICT_SCHEMA = {
    "name": "verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"agree": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["agree", "reason"],
        "additionalProperties": False,
    },
}


@dataclass
class Case:
    id: str
    input: Any
    checks: list[Check]


def lesson_names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(lessons.__path__) if m.ispkg)


def resolve(prefix: str) -> str:
    matches = [name for name in lesson_names() if name.startswith(prefix)]
    if len(matches) != 1:
        raise SystemExit(f"no single lesson for {prefix!r}, have: {lesson_names()}")
    return matches[0]


def implementations(lesson: str) -> list[str]:
    folder = ROOT / "lessons" / lesson
    return sorted(p.stem for p in folder.glob("with_*.py")) + ["hard_way"]


def _module_path(lesson: str, impl: str) -> Path:
    return ROOT / "lessons" / lesson / f"{impl}.py"


def lines_of_code(source: str) -> int:
    skipped = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }  # fmt: skip
    lines: set[int] = set()
    for token in tokenize.tokenize(io.BytesIO(source.encode()).readline):
        if token.type not in skipped:
            lines.update(range(token.start[0], token.end[0] + 1))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.difference_update(range(first.lineno, first.end_lineno + 1))
    return len(lines)


def third_party_imports(source: str) -> list[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots - set(sys.stdlib_module_names) - {"lessons", "evals"})


def import_cost(lesson: str, impl: str) -> dict:
    script = (
        "import json, sys, time\n"
        "start = time.perf_counter()\n"
        f"import lessons.{lesson}.{impl}\n"
        "print(json.dumps({'import_seconds': time.perf_counter() - start, 'modules': len(sys.modules)}))"
    )
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", script], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.splitlines()[-1])


def complexity(lesson: str, impl: str) -> dict:
    source = _module_path(lesson, impl).read_text()
    return {"loc": lines_of_code(source), "deps": third_party_imports(source), **import_cost(lesson, impl)}


def run_case(run: Callable[[Any], Any], case: Case) -> dict:
    start = time.perf_counter()
    output, error = None, None
    with capture() as trace:
        try:
            output = run(case.input)
        except NotImplementedError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    passed = {}
    for check in case.checks:
        try:
            passed[check.name] = error is None and bool(check.fn(output, trace))
        except Exception:
            passed[check.name] = False
    return {"output": output, "error": error, "seconds": seconds, "checks": passed, **asdict(trace)}


def evaluate(lesson: str, impl: str) -> dict | None:
    """Returns None while the implementation is still a stub."""
    problem = importlib.import_module(f"lessons.{lesson}.problem")
    module = importlib.import_module(f"lessons.{lesson}.{impl}")
    try:
        cases = {case.id: run_case(module.run, case) for case in problem.CASES}
    except NotImplementedError:
        return None
    return {"cases": cases, "complexity": complexity(lesson, impl)}


def _result_path(lesson: str, impl: str) -> Path:
    return RESULTS / lesson / f"{impl}.json"


def load_or_run(lesson: str, impl: str, fresh: bool) -> dict | None:
    path = _result_path(lesson, impl)
    if impl != "hard_way" and path.exists() and not fresh:
        return json.loads(path.read_text())
    result = evaluate(lesson, impl)
    if result is None:
        return None
    if all(not case["error"] and all(case["checks"].values()) for case in result["cases"].values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
    return result


def judge(question: Any, reference: Any, candidate: Any) -> tuple[bool, str]:
    reply = OpenAI().chat.completions.create(
        model=lessons.MODEL,
        response_format={"type": "json_schema", "json_schema": VERDICT_SCHEMA},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Input: {question}\n\nAnswer A: {reference}\n\nAnswer B: {candidate}"},
        ],
    )
    verdict = json.loads(reply.choices[0].message.content)
    return verdict["agree"], verdict["reason"]


def agreement(
    question: Any,
    reference: Any,
    candidate: Any,
    judge_fn: Callable[[Any, Any, Any], tuple[bool, str]] = judge,
) -> tuple[bool, str]:
    if candidate is None:
        return False, "no output"
    if reference == candidate:
        return True, "identical"
    if isinstance(reference, dict) and isinstance(candidate, dict):
        same = sum(reference.get(key) == candidate.get(key) for key in reference)
        return same == len(reference), f"{same}/{len(reference)} fields match"
    return judge_fn(question, reference, candidate)


def _identity(output: Any) -> Any:
    return output


def compare(problem: ModuleType, candidate: dict, reference: dict) -> list[dict]:
    """Case-by-case agreement plus whether both used the same tools."""
    comparable = getattr(problem, "comparable", _identity)
    rows = []
    for case in problem.CASES:
        ours, theirs = candidate["cases"][case.id], reference["cases"][case.id]
        agree, reason = agreement(case.input, comparable(theirs["output"]), comparable(ours["output"]))
        tools = {call["name"] for call in ours["tool_calls"]} == {call["name"] for call in theirs["tool_calls"]}
        rows.append({"case": case.id, "agree": agree, "same_tools": tools, "reason": reason})
    return rows


def summarize(result: dict) -> dict:
    cases = result["cases"].values()
    checks = [ok for case in cases for ok in case["checks"].values()]
    return {
        "checks": f"{sum(checks)}/{len(checks)}",
        "llm": sum(case["llm_calls"] for case in cases),
        "in_tok": sum(case["input_tokens"] for case in cases),
        "out_tok": sum(case["output_tokens"] for case in cases),
        "embed_tok": sum(case["embedding_tokens"] for case in cases),
        "sec": f"{sum(case['seconds'] for case in cases):.1f}",
        "loc": result["complexity"]["loc"],
        "deps": len(result["complexity"]["deps"]),
        "import_s": f"{result['complexity']['import_seconds']:.2f}",
        "modules": result["complexity"]["modules"],
    }


def table(rows: list[dict]) -> str:
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(row[h])) for row in rows)) for h in headers}
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines += ["  ".join(str(row[h]).ljust(widths[h]) for h in headers) for row in rows]
    return "\n".join(lines)


def report(lesson: str, results: dict[str, dict | None]) -> str:
    problem = importlib.import_module(f"lessons.{lesson}.problem")
    rows = [{"impl": impl, **summarize(result)} for impl, result in results.items() if result]
    sections = [lesson, table(rows) if rows else "no results"]
    failures = [
        f"  {impl} / {case_id}: {case['error'] or [name for name, ok in case['checks'].items() if not ok]}"
        for impl, result in results.items() if result
        for case_id, case in result["cases"].items()
        if case["error"] or not all(case["checks"].values())
    ]  # fmt: skip
    if failures:
        sections += ["failures:", *failures]
    candidate = results.get("hard_way")
    if candidate is None:
        return "\n".join([*sections, "hard_way: not implemented yet"])
    for impl, reference in results.items():
        if impl != "hard_way" and reference:
            sections += [f"hard_way vs {impl}:", table(compare(problem, candidate, reference))]
    return "\n".join(sections)
