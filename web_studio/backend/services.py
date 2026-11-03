"""FoundryServices: Web Studio バックエンドの DI コンテナ。

設定・DB・LLM クライアント・各リポジトリ／エンジンのファクトリ・ジョブマネージャ・WebSocket ハブ・
内蔵モック ComfyUI を保持し、設定更新時には LLM クライアントを再構築する。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from core.asset_publisher.knowledge_graph import KnowledgeGraph
from core.asset_publisher.matrix_transpiler import ContentMatrixTranspiler
from core.asset_publisher.repository import ArticleRepository
from core.asset_publisher.writer_pipeline import ArticleWriterPipeline, WriterConfig
from core.common import utc_now_iso
from core.config import FoundrySettings
from core.db import Database
from core.llm_client import HarnessConfig, LLMClient, create_llm_client
from core.manga_pipeline.comfy_client import ComfyClient
from core.manga_pipeline.comfy_mock_server import MockComfyServer
from core.manga_pipeline.storyboard_generator import StoryboardConfig, StoryboardGenerator
from core.structure_engine.extractor import StructureExtractor
from core.structure_engine.repository import StructureRepository
from core.structure_engine.schemas import UniversalStructure
from core.structure_engine.seeds.loader import SEED_FILES, load_seed_file, seed_database, seed_dir
from core.structure_engine.transpiler import StructureTranspiler
from core.style_alchemy.dataset_curator import DatasetCurator
from core.style_alchemy.grid_laboratory import GridLaboratory
from core.style_alchemy.repository import StyleLabRepository
from web_studio.backend.jobs import JobManager
from web_studio.backend.stores import ResearchLogStore, StoryboardStore
from web_studio.backend.ws_hub import WebSocketHub

__all__ = ["FoundryServices", "deep_merge"]

logger = logging.getLogger("caf.backend.services")

_UPDATABLE_SECTIONS = ("llm", "comfy")


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """辞書の再帰マージ（patch 側が優先、dict 同士は再帰）。"""
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class FoundryServices:
    """バックエンド全体で共有するサービス群。"""

    def __init__(self, settings: FoundrySettings) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.db = Database(settings.db_path)
        self.hub = WebSocketHub()
        self.jobs = JobManager(self.db, self.hub)
        self.structure_repo = StructureRepository(self.db)
        self.style_repo = StyleLabRepository(self.db)
        self.article_repo = ArticleRepository(self.db)
        self.storyboards = StoryboardStore(self.db)
        self.research_logs = ResearchLogStore(self.db)
        self.llm_client: LLMClient = create_llm_client(settings.llm, self.db)
        self.harness_config = HarnessConfig.from_settings(settings.llm.harness)
        self.started_at = utc_now_iso()
        self.seed_report: dict[str, Any] | None = None
        self._mock_comfy: MockComfyServer | None = None
        self._mock_lock = threading.Lock()

    # -- 起動・終了 -------------------------------------------------------------

    def initialize(self) -> None:
        """起動時初期化: 構造が0件なら存在するシードファイルを投入する。"""
        if self.structure_repo.count() == 0:
            report = self.load_seeds(overwrite=False)
            logger.info("初期シードを投入しました: %s", report)

    async def shutdown(self) -> None:
        """ジョブ停止・LLM 資源解放・内蔵モック ComfyUI 停止。"""
        await self.jobs.shutdown()
        with contextlib.suppress(Exception):
            await self.llm_client.aclose()
        await asyncio.to_thread(self.stop_mock_comfy)

    # -- シード -----------------------------------------------------------------

    def available_seed_structures(self) -> tuple[list[UniversalStructure], list[str], list[str]]:
        """存在し読み込めるシードファイルの全構造、欠落ファイル、読込失敗ファイルを返す。"""
        structures: list[UniversalStructure] = []
        missing: list[str] = []
        invalid: list[str] = []
        for filename in SEED_FILES.values():
            path = seed_dir() / filename
            if not path.exists():
                logger.warning("シードファイルが見つからないためスキップします: %s", path)
                missing.append(filename)
                continue
            try:
                structures.extend(load_seed_file(filename))
            except (ValueError, ValidationError, json.JSONDecodeError, OSError) as exc:
                logger.warning("シードファイルを読み込めないためスキップします: %s (%s)", path, exc)
                invalid.append(filename)
        return structures, missing, invalid

    def load_seeds(self, overwrite: bool = False) -> dict[str, Any]:
        structures, missing, invalid = self.available_seed_structures()
        report = seed_database(self.structure_repo, overwrite=overwrite, structures=structures)
        result = {**report.model_dump(mode="json"), "missing_files": missing, "invalid_files": invalid}
        self.seed_report = result
        return result

    def seed_summary(self) -> dict[str, Any]:
        by_category = {
            category: self.structure_repo.count(category, source_kind="seed") for category in SEED_FILES
        }
        return {
            "total": self.structure_repo.count(source_kind="seed"),
            "by_category": by_category,
            "files": {category: (seed_dir() / filename).exists() for category, filename in SEED_FILES.items()},
        }

    # -- エンジン ファクトリ ------------------------------------------------------

    def extractor(self) -> StructureExtractor:
        return StructureExtractor(self.llm_client, self.harness_config)

    def transpiler(self) -> StructureTranspiler:
        return StructureTranspiler(self.llm_client, self.harness_config)

    def matrix_transpiler(self) -> ContentMatrixTranspiler:
        return ContentMatrixTranspiler(self.llm_client, self.harness_config)

    def writer(self, config: WriterConfig | None = None) -> ArticleWriterPipeline:
        return ArticleWriterPipeline(self.llm_client, config or WriterConfig(), self.harness_config)

    def storyboard_generator(self, config: StoryboardConfig) -> StoryboardGenerator:
        return StoryboardGenerator(config, llm_client=self.llm_client)

    def grid_lab(self) -> GridLaboratory:
        return GridLaboratory(self.style_repo)

    def dataset_curator(self, trigger_token: str, use_llm: bool) -> DatasetCurator:
        return DatasetCurator(
            trigger_token=trigger_token,
            llm_client=self.llm_client if use_llm else None,
            harness_config=self.harness_config,
        )

    def knowledge_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(self.db).build_from_db(self.db)

    # -- LLM ---------------------------------------------------------------------

    def llm_status(self) -> dict[str, Any]:
        client = self.llm_client
        return {
            "provider": client.effective_provider,
            "configured_provider": client.provider,
            "fallback_active": client.fallback_active,
            "model": client.model,
        }

    async def reload_settings(self, update: Mapping[str, Any]) -> FoundrySettings:
        """設定を部分更新（深いマージ→検証→保存）し、LLM クライアントを再構築する。"""
        if not isinstance(update, Mapping):
            raise ValueError("設定更新は {\"llm\": {...}, \"comfy\": {...}} 形式のオブジェクトで指定してください")
        unknown = sorted(set(update) - set(_UPDATABLE_SECTIONS))
        if unknown:
            raise ValueError(f"更新できない設定セクションです: {', '.join(unknown)}（llm / comfy のみ）")
        patch: dict[str, Any] = {}
        for section in _UPDATABLE_SECTIONS:
            if section not in update:
                continue
            value = update[section]
            if not isinstance(value, Mapping):
                raise ValueError(f"{section} はオブジェクトで指定してください")
            patch[section] = dict(value)
        llm_patch = patch.get("llm")
        if llm_patch is not None and "***" in str(llm_patch.get("api_key", "")):
            llm_patch.pop("api_key")
        current = self.settings.model_dump(mode="json")
        current.pop("data_dir", None)
        merged = deep_merge(current, patch)
        merged["data_dir"] = str(self.settings.data_dir)
        new_settings = FoundrySettings.model_validate(merged)
        new_settings.save()
        previous_client = self.llm_client
        previous_mock = self.settings.comfy.use_mock
        self.settings = new_settings
        self.llm_client = create_llm_client(new_settings.llm, self.db)
        self.harness_config = HarnessConfig.from_settings(new_settings.llm.harness)
        with contextlib.suppress(Exception):
            await previous_client.aclose()
        if previous_mock and not new_settings.comfy.use_mock:
            await asyncio.to_thread(self.stop_mock_comfy)
        return new_settings

    # -- ComfyUI -----------------------------------------------------------------

    def ensure_mock_comfy(self) -> MockComfyServer:
        """内蔵モック ComfyUI を遅延起動する（スレッドセーフ）。"""
        with self._mock_lock:
            if self._mock_comfy is None:
                self._mock_comfy = MockComfyServer().start()
            return self._mock_comfy

    def stop_mock_comfy(self) -> None:
        with self._mock_lock:
            if self._mock_comfy is not None:
                self._mock_comfy.stop()
                self._mock_comfy = None

    @property
    def mock_comfy_running(self) -> bool:
        return self._mock_comfy is not None

    async def comfy_base_url(self) -> str:
        if self.settings.comfy.use_mock:
            server = await asyncio.to_thread(self.ensure_mock_comfy)
            return server.base_url
        return self.settings.comfy.base_url

    def comfy_client(self, base_url: str, timeout_s: float | None = None) -> ComfyClient:
        return ComfyClient(
            base_url,
            client_id=f"{self.settings.comfy.client_id_prefix}-{uuid.uuid4().hex[:10]}",
            timeout_s=timeout_s if timeout_s is not None else self.settings.comfy.timeout_s,
        )
