"""初期プリセット「普遍的構造データベース」（物語原型・漫画演出リズム・説得構造 各20パターン）。

``authoring`` サブパッケージ（JSON 再生成用の著作スクリプト）はここでは import しない。
"""

from __future__ import annotations

from core.structure_engine.seeds.loader import (
    SEED_FILES,
    SEED_SCHEMA_VERSION,
    SeedReport,
    load_all_seeds,
    load_seed_file,
    seed_database,
    seed_dir,
)

__all__ = [
    "SEED_FILES",
    "SEED_SCHEMA_VERSION",
    "SeedReport",
    "load_all_seeds",
    "load_seed_file",
    "seed_database",
    "seed_dir",
]
