"""
Chinese Semantic Relations — zero-LLM bilingual semantic relation analyzer.

Analyzes Chinese and English text to extract 38 semantic relation types
across 8 branches, generate targeted challenge strategies for gaps, and
score semantic quality — all without LLM calls.

Quick start::

    from chinese_semantic_relations import SemanticRelationEngine

    engine = SemanticRelationEngine()

    # Analyze text and find semantic gaps
    plan = engine.plan_challenge(
        "我们的AI平台帮助所有学生提高成绩",
        difficulty_level=2,
    )
    print(plan.instruction)          # targeted probe question
    print(plan.probes[0].gap_description_zh)  # why this probe

    # Score semantic quality
    score = engine.evaluate_response("因为我们通过个性化推荐...")
    print(score.total)               # 0-25 scale
    print(score.recommendations)     # improvement suggestions

    # Multi-round conversation analysis
    trajectory = engine.analyze_conversation([
        {"user_response": "因为市场需求大所以值得做"},
        {"user_response": "具体来说聚焦在教育行业"},
    ])
    print(trajectory.growth)         # semantic growth over rounds

Architecture:
  - ontology.py    — 8-branch relation type taxonomy (38 types)
  - analyzer.py    — zero-LLM regex-based relation extraction
  - strategies.py  — gap-to-challenge strategy mapping
  - evaluator.py   — semantic quality scoring
  - __init__.py    — SemanticRelationEngine facade (this file)

All operations are zero-LLM (regex + rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ontology import (
    RelationBranch,
    RelationType,
    RelationSpec,
    RELATION_ONTOLOGY,
    BRANCH_DISPLAY,
    get_branch_relations,
    get_high_relevance_relations,
    get_relations_for_dimension,
    get_relations_for_bloom,
)
from .analyzer import (
    SemanticRelationAnalyzer,
    RelationProfile,
    DetectedRelation,
    ConversationSemanticTrajectory,
)
from .strategies import (
    SemanticChallengeStrategyMapper,
    SemanticStrategyPlan,
    SemanticProbe,
    PersonaResolver,
)
from .evaluator import (
    SemanticResponseEvaluator,
    SemanticScore,
)


@dataclass
class StrategyPlan:
    """Lightweight strategy plan contract for interop with external systems."""

    strategy_key: str
    instruction: str
    dimension_affinity: list[str] = field(default_factory=list)
    confidence: float = 0.5
    fallback_instruction: str = "请基于用户回答进行追问或反驳。"


class SemanticRelationEngine:
    """Facade that composes analyzer, strategy mapper, and evaluator.

    Single entry point for all semantic relation operations.

    Args:
        persona_resolver: Optional callable ``(dims: tuple[str, ...]) -> list[str]``
            that maps dimension affinity tags to persona/role keys. Useful for
            AI training platforms that have their own role system.
    """

    def __init__(
        self,
        *,
        persona_resolver: PersonaResolver | None = None,
    ) -> None:
        self._analyzer = SemanticRelationAnalyzer()
        self._mapper = SemanticChallengeStrategyMapper(persona_resolver=persona_resolver)
        self._evaluator = SemanticResponseEvaluator()

    # -- Health check -------------------------------------------------------

    def health_check(self) -> dict:
        return {
            "ok": True,
            "engine": "SemanticRelationEngine",
            "relation_types": len(RELATION_ONTOLOGY),
            "branches": len(RelationBranch),
        }

    # -- Analysis -----------------------------------------------------------

    def analyze(self, text: str) -> RelationProfile:
        """Extract semantic relations from a single text."""
        return self._analyzer.analyze(text)

    def analyze_conversation(self, rounds: list[dict]) -> ConversationSemanticTrajectory:
        """Per-round profiles, merged profile, growth stats, and cross-round hints."""
        return self._analyzer.analyze_conversation(rounds)

    # -- Challenge Strategy -------------------------------------------------

    def plan_challenge(
        self,
        user_text: str,
        *,
        difficulty_level: int = 2,
        target_dimension: str | None = None,
        expected_types: set[RelationType] | None = None,
        sector: str | None = None,
    ) -> SemanticStrategyPlan:
        """Analyze text and generate a targeted challenge strategy."""
        profile = self._analyzer.analyze(user_text)
        return self._mapper.plan_from_profile(
            profile,
            difficulty_level=difficulty_level,
            target_dimension=target_dimension,
            expected_types=expected_types,
            sector=sector,
        )

    def plan_from_profile(
        self,
        profile: RelationProfile,
        *,
        difficulty_level: int = 2,
        target_dimension: str | None = None,
        sector: str | None = None,
    ) -> SemanticStrategyPlan:
        """Generate strategy from a pre-computed profile."""
        return self._mapper.plan_from_profile(
            profile,
            difficulty_level=difficulty_level,
            target_dimension=target_dimension,
            sector=sector,
        )

    # -- Evaluation ---------------------------------------------------------

    def evaluate_response(self, text: str) -> SemanticScore:
        """Evaluate semantic quality of a user response."""
        profile = self._analyzer.analyze(text)
        return self._evaluator.evaluate(profile)

    def evaluate_profile(self, profile: RelationProfile) -> SemanticScore:
        """Evaluate semantic quality from a pre-computed profile."""
        return self._evaluator.evaluate(profile)

    # -- Conversion ---------------------------------------------------------

    def to_strategy_plan(self, plan: SemanticStrategyPlan) -> StrategyPlan:
        """Convert to lightweight StrategyPlan for external system interop."""
        return StrategyPlan(
            strategy_key=plan.strategy_key,
            instruction=plan.instruction,
            dimension_affinity=plan.dimension_affinity,
            confidence=plan.confidence,
            fallback_instruction=plan.fallback_instruction,
        )


__all__ = [
    "SemanticRelationEngine",
    "SemanticRelationAnalyzer",
    "SemanticChallengeStrategyMapper",
    "SemanticResponseEvaluator",
    "RelationProfile",
    "ConversationSemanticTrajectory",
    "DetectedRelation",
    "SemanticStrategyPlan",
    "SemanticProbe",
    "SemanticScore",
    "StrategyPlan",
    "PersonaResolver",
    "RelationType",
    "RelationBranch",
    "RelationSpec",
    "RELATION_ONTOLOGY",
    "BRANCH_DISPLAY",
    "get_branch_relations",
    "get_high_relevance_relations",
    "get_relations_for_dimension",
    "get_relations_for_bloom",
]
