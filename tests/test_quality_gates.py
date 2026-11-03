"""品質ゲート: Zero Placeholder Policy の機械検査と全モジュールの import 健全性検査。"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

# 自己一致を避けるため、禁止語は連結で組み立てる。
_FORBIDDEN_TOKENS = [
    "TO" + "DO",
    "FIX" + "ME",
    "Not" + "ImplementedError",
    "以下" + "同様",
    "ダミー" + "実装",
    "（" + "省略" + "）",
    "(" + "省略" + ")",
    "省略" + "します",
    "中" + "略",
]
_FORBIDDEN_LINE_PATTERNS = [
    re.compile(r"^\s*" + "pa" + r"ss\s*(#.*)?$"),
    re.compile(r"^\s*\.\.\.\s*(#.*)?$"),
]

_SCAN_GLOBS = [
    ("core", "**/*.py"),
    ("web_studio/backend", "**/*.py"),
    ("tests", "**/*.py"),
    ("web_studio/frontend/src", "**/*.ts"),
    ("web_studio/frontend/src", "**/*.tsx"),
]


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for base, pattern in _SCAN_GLOBS:
        base_dir = ROOT / base
        if not base_dir.exists():
            continue
        for path in base_dir.glob(pattern):
            if "node_modules" in path.parts or "__pycache__" in path.parts:
                continue
            files.append(path.resolve())
    runner = ROOT / "run_studio.py"
    if runner.exists():
        files.append(runner.resolve())
    return sorted(set(files))


def test_no_placeholder_patterns() -> None:
    violations: list[str] = []
    for path in _iter_source_files():
        if path == THIS_FILE:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for token in _FORBIDDEN_TOKENS:
                if token in line:
                    violations.append(f"{path.relative_to(ROOT)}:{lineno}: 禁止語 {token!r}")
            if path.suffix == ".py":
                for pattern in _FORBIDDEN_LINE_PATTERNS:
                    if pattern.match(line):
                        violations.append(f"{path.relative_to(ROOT)}:{lineno}: 空実装行 {line.strip()!r}")
    assert not violations, "Zero Placeholder Policy 違反:\n" + "\n".join(violations)


def _iter_module_names() -> list[str]:
    names: list[str] = []
    for package_name in ("core", "web_studio"):
        package_dir = ROOT / package_name
        if not package_dir.exists():
            continue
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _iter_module_names())
def test_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)
