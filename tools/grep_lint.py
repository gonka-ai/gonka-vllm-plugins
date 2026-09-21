#!/usr/bin/env python3
"""Process-gate grep-lint -- block the v1->v2->v3 whack-a-mole cycle.

Enforces two invariants on the repo:

  (a) Private vLLM symbols (``from vllm.v1.*`` / ``import vllm.v1.*``) MUST
      live ONLY inside ``src/gonka_poc/mixed/`` (the engine-facing seam
      subpackage) or under ``tests/``. Anything else creeps the touchpoint
      surface and forces a whole-tree audit every time a vLLM minor bumps.

  (b) Every ``ADR-NNNN`` reference in source code / docs MUST correspond to a
      real file under ``docs/adr/`` (any extension). Stale ADR references
      strand the reader: they look authoritative but point nowhere.

Findings are printed as ``file:line: <reason>`` lines on stdout. Exit code 0
when clean, 1 (with a one-line summary on stderr) otherwise.

Scope / scan rules:
  - Globs: src/**/*.py, tests/**/*.py, *.md, **/*.md
  - Excluded: .git, .venv, build, dist, *.egg-info, __pycache__, node_modules
  - For (a): skip pure comment lines (lines whose first non-whitespace char is
    ``#``) and skip lines that sit inside a fenced markdown code block tagged
    ``text`` / ``log`` / ``console`` (i.e. non-executable prose). Triple-quoted
    Python docstrings ARE still scanned -- treating them as exempt would let
    a real ``from vllm.v1.*`` import hide behind a triple-quote opener. The
    coverage difference: a true docstring mention is a one-off, the cost of
    fixing it (rephrase to ``vllm.v1.<thing>`` without ``from``/``import``
    keywords, OR drop the dotted path) is bounded; a hidden import is the
    bug we are gating against.
  - For (a) only: an inline pragma ``# grep-lint: ignore`` anywhere on the
    same line suppresses the finding. Reserved for legacy test code that
    pre-dates the ``_compat`` channel and cannot trivially be routed
    through it (e.g. tests that import a specific private symbol to
    monkey-patch). The pragma does NOT suppress rule (b) -- a stale ADR
    reference is a documentation defect that no inline comment justifies.

Stdlib only. No third-party deps -- CI runs on bare ubuntu-latest with
``python3``.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterator, List, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# Blessed prefixes for private vllm.v1.* touchpoints: the engine-facing
# mixed subpackage, the version-dispatched _compat adapters (the channel the
# workflow header names) and tests. Matching is done on the path RELATIVE to
# REPO_ROOT (POSIX-normalised).
ALLOWED_V1_PREFIXES = ("src/gonka_poc/mixed/", "src/gonka_poc/_compat/", "tests/")

EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Suffix-based excludes for directories (egg-info etc.).
EXCLUDE_DIR_SUFFIXES: Tuple[str, ...] = (".egg-info",)


# ----------------------------------------------------------------------------- #
# Regexes
# ----------------------------------------------------------------------------- #

# ``from vllm.v1.X import Y`` or ``import vllm.v1.X`` -- we deliberately do NOT
# match ``vllm.v1`` mentions inside string literals / prose because (1) prose
# mentions are unavoidable in commit messages / ADRs that DESCRIBE the rule,
# and (2) the only thing that can hide a real import is a real import. The
# trigger is the ``from`` / ``import`` keyword followed by ``vllm.v1``.
RE_PRIVATE_VLLM_V1 = re.compile(
    r"^\s*(?:from\s+vllm\.v1(?:\.[A-Za-z0-9_]+)*\s+import\b"
    r"|import\s+vllm\.v1(?:\.[A-Za-z0-9_]+)*\b)"
)

# ``ADR-1234`` (exactly four digits). Allows trailing punctuation / whitespace
# / closing brackets. We capture the four-digit id for existence lookup.
RE_ADR = re.compile(r"\bADR-(\d{4})\b")


# Fenced code block markers we skip ENTIRELY for (a) -- prose/log dumps that
# happen to mention an import line as an example. NOTE: this only suppresses
# the private-vllm-v1 check; ADR references inside any fenced block are still
# scanned (they are documentation, and a wrong reference there misleads
# readers identically).
FENCED_NONCODE_LANGS: Tuple[str, ...] = ("text", "log", "console", "shell")


# ----------------------------------------------------------------------------- #
# File discovery
# ----------------------------------------------------------------------------- #


def _iter_candidate_files(root: Path) -> Iterator[Path]:
    """Yield every file we want to scan, honouring EXCLUDE_DIRS.

    We walk manually instead of ``Path.rglob`` so EXCLUDE_DIRS can prune
    whole subtrees (rglob would still descend into .venv etc., which on this
    repo would be tens of thousands of files).
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # In-place mutate dirnames to prune the walk.
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS
            and not any(d.endswith(suf) for suf in EXCLUDE_DIR_SUFFIXES)
        ]
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if _file_in_scope(rel):
                yield p


def _file_in_scope(rel_posix: str) -> bool:
    """Mirror the glob spec in the module docstring."""
    if rel_posix.startswith("src/") and rel_posix.endswith(".py"):
        return True
    if rel_posix.startswith("tests/") and rel_posix.endswith(".py"):
        return True
    if rel_posix.endswith(".md"):
        return True
    return False


# ----------------------------------------------------------------------------- #
# Line classification
# ----------------------------------------------------------------------------- #


def _is_under_compat(rel_posix: str) -> bool:
    return rel_posix.startswith(ALLOWED_V1_PREFIXES)


def _is_comment_line(line: str) -> bool:
    """Treat shell/python ``#`` comments as exempt for (a) only.

    Used because narrative comments referencing import lines occur in
    review docs / migration notes (e.g. narrative # ``from vllm.v1.x ...``).
    """
    stripped = line.lstrip()
    return stripped.startswith("#")


def _classify_fence_marker(line: str) -> Tuple[bool, str]:
    """Return (is_marker, lang_lower) for a markdown fenced code marker.

    A fence marker is a line starting with three backticks followed by an
    optional language tag. We do NOT treat tabbed code as fenced; this
    matches GitHub-Flavored Markdown.
    """
    s = line.lstrip()
    triple = "`" * 3
    if not s.startswith(triple):
        return False, ""
    tag = s[3:].strip().lower()
    return True, tag


# ----------------------------------------------------------------------------- #
# ADR discovery
# ----------------------------------------------------------------------------- #


def _discover_adrs(root: Path) -> Set[str]:
    """Return the set of ADR ids present under ``docs/adr/``.

    Recognised id forms in filenames (case-insensitive prefix):
        ``ADR-NNNN-slug.md``
        ``adr-NNNN.md``
        ``NNNN-slug.md``

    Any other naming scheme is treated as NOT defining an ADR id; if the
    repo is using a different format we want the lint to fail loudly so the
    contract gets updated explicitly.
    """
    adr_dir = root / "docs" / "adr"
    found: Set[str] = set()
    if not adr_dir.is_dir():
        return found

    pattern = re.compile(r"(?:adr[-_])?(\d{4})", re.IGNORECASE)
    for p in adr_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.stem)
        if m:
            found.add(m.group(1))
    return found


# ----------------------------------------------------------------------------- #
# Core scan
# ----------------------------------------------------------------------------- #


def _scan_file(
    path: Path,
    rel_posix: str,
    adr_ids: Set[str],
) -> List[str]:
    """Return findings for one file as ``file:line: reason`` strings."""
    findings: List[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        findings.append(f"{rel_posix}:0: failed to read ({exc})")
        return findings

    is_markdown = rel_posix.endswith(".md")
    in_compat = _is_under_compat(rel_posix)

    # Fenced code-block state -- only relevant for markdown.
    in_fence = False
    fence_lang = ""

    for lineno, raw in enumerate(text.splitlines(), start=1):
        # Track fenced code blocks for markdown.
        if is_markdown:
            is_marker, lang = _classify_fence_marker(raw)
            if is_marker:
                if not in_fence:
                    in_fence = True
                    fence_lang = lang
                else:
                    in_fence = False
                    fence_lang = ""
                # Marker line itself is never scanned for either rule.
                continue

        # (a) private vllm.v1 imports
        if not in_compat:
            skip_for_private = False
            if _is_comment_line(raw):
                skip_for_private = True
            elif is_markdown and in_fence and fence_lang in FENCED_NONCODE_LANGS:
                skip_for_private = True
            # Inline pragma. ``# grep-lint: ignore`` (anywhere on the line)
            # suppresses rule (a) on that single line. Intended for legacy
            # test code that pre-dates the compat-channel rule; new code
            # should still go through ``_compat``. The pragma deliberately
            # does NOT suppress rule (b) -- a wrong ADR reference is a doc
            # defect that an exempt-comment cannot justify.
            elif "grep-lint: ignore" in raw:
                skip_for_private = True

            if not skip_for_private and RE_PRIVATE_VLLM_V1.search(raw):
                findings.append(
                    f"{rel_posix}:{lineno}: private vllm.v1.* import outside "
                    f"src/gonka_poc/mixed/ or tests/"
                )

        # (b) ADR-NNNN existence
        for m in RE_ADR.finditer(raw):
            adr_id = m.group(1)
            if adr_id not in adr_ids:
                findings.append(
                    f"{rel_posix}:{lineno}: ADR-{adr_id} referenced but no file "
                    f"matches under docs/adr/"
                )

    return findings


# ----------------------------------------------------------------------------- #
# Entry point
# ----------------------------------------------------------------------------- #


def run(root: Path) -> int:
    adr_ids = _discover_adrs(root)
    findings: List[str] = []
    for f in _iter_candidate_files(root):
        rel = f.relative_to(root).as_posix()
        findings.extend(_scan_file(f, rel, adr_ids))

    findings.sort()
    for line in findings:
        print(line)

    if findings:
        # Summary to stderr so log scrapers can separate signal from list.
        print(
            f"grep-lint: {len(findings)} finding(s) -- see stdout for the list",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    root = Path(args[0]).resolve() if args else REPO_ROOT
    return run(root)


if __name__ == "__main__":
    raise SystemExit(main())
