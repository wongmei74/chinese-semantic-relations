"""
Semantic Relation Ontology — 8-branch taxonomy with 38 relation types.

Defines a formal semantic relation type system with properties derived from
WordNet, FrameNet, and ConceptNet. Each relation type carries:
  - Formal logic properties (symmetric, transitive)
  - Relevance weights for product/business reasoning assessment
  - Bloom's taxonomy alignment (bloom_level_min)
  - Dimension affinity tags for quality evaluation

Design principles:
  - Each RelationType carries formal logic properties (symmetric, transitive)
  - product_relevance weights focus on relations most valuable for
    structured reasoning (causal > evaluative)
  - bloom_level_min aligns with Bloom's cognitive taxonomy
  - dimension_affinity maps to quality assessment dimensions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RelationBranch(str, Enum):
    EQUIVALENCE = "equivalence"
    HIERARCHY = "hierarchy"
    APPROXIMATION = "approximation"
    SEMANTIC_DISTANCE = "semantic_distance"
    OPPOSITION = "opposition"
    TEMPORAL = "temporal"
    FUNCTIONAL = "functional"
    EVALUATIVE = "evaluative"


class RelationType(str, Enum):
    # 1. Equivalence
    STRICT_SYNONYM = "strict_synonym"
    TERMINOLOGICAL_SYNONYM = "terminological_synonym"
    LOGICAL_EQUIVALENCE = "logical_equivalence"
    COREFERENCE = "coreference"

    # 2. Hierarchy
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    INSTANCE_OF = "instance_of"
    PART_OF = "part_of"
    MEMBER_OF = "member_of"

    # 3. Approximation
    NEAR_SYNONYM = "near_synonym"
    ANALOGY = "analogy"
    METAPHOR = "metaphor"
    POLYSEMY = "polysemy"

    # 4. Semantic Distance
    SAME_FIELD = "same_field"
    FRAME_BASED = "frame_based"
    COLLOCATION = "collocation"
    THEMATIC = "thematic"

    # 5. Opposition
    COMPLEMENTARY_ANTONYM = "complementary_antonym"
    GRADABLE_ANTONYM = "gradable_antonym"
    CONVERSE = "converse"
    DIRECTIONAL_ANTONYM = "directional_antonym"
    CONTRADICTION = "contradiction"

    # 6. Temporal / Sequential
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSE_EFFECT = "cause_effect"
    CONDITION_CONSEQUENCE = "condition_consequence"
    PURPOSE_MEANS = "purpose_means"
    CAUSATIVE = "causative"
    PROCESS_STAGE = "process_stage"

    # 7. Functional
    TOOL_PURPOSE = "tool_purpose"
    AGENT_ACTION = "agent_action"
    ACTION_RESULT = "action_result"
    ACTION_PATIENT = "action_patient"
    FUNCTIONAL_EQUIVALENCE = "functional_equivalence"

    # 8. Evaluative
    POSITIVE_CONNOTATION = "positive_connotation"
    NEGATIVE_CONNOTATION = "negative_connotation"
    EUPHEMISM = "euphemism"
    VAGUE_QUANTIFIER = "vague_quantifier"

    # 9. Evidence
    EVIDENCE_CITATION = "evidence_citation"


@dataclass(frozen=True)
class RelationSpec:
    """Full specification for a semantic relation type."""

    type: RelationType
    branch: RelationBranch
    label_zh: str
    label_en: str

    is_symmetric: bool = False
    is_transitive: bool = False

    product_relevance: float = 0.5
    bloom_level_min: int = 2
    dimension_affinity: tuple[str, ...] = ()

    wordnet_relation: str | None = None
    conceptnet_relation: str | None = None


# ---------------------------------------------------------------------------
# Full ontology registry
# ---------------------------------------------------------------------------

RELATION_ONTOLOGY: dict[RelationType, RelationSpec] = {
    # ── 1. Equivalence ────────────────────────────────────────────────────────
    RelationType.STRICT_SYNONYM: RelationSpec(
        type=RelationType.STRICT_SYNONYM,
        branch=RelationBranch.EQUIVALENCE,
        label_zh="完全同义",
        label_en="Strict Synonym",
        is_symmetric=True, is_transitive=True,
        product_relevance=0.4, bloom_level_min=2,
        dimension_affinity=("consistency",),
        wordnet_relation="synonym", conceptnet_relation="Synonym",
    ),
    RelationType.TERMINOLOGICAL_SYNONYM: RelationSpec(
        type=RelationType.TERMINOLOGICAL_SYNONYM,
        branch=RelationBranch.EQUIVALENCE,
        label_zh="术语同义",
        label_en="Terminological Synonym",
        is_symmetric=True, is_transitive=True,
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("structure",),
    ),
    RelationType.LOGICAL_EQUIVALENCE: RelationSpec(
        type=RelationType.LOGICAL_EQUIVALENCE,
        branch=RelationBranch.EQUIVALENCE,
        label_zh="逻辑等价",
        label_en="Logical Equivalence",
        is_symmetric=True, is_transitive=True,
        product_relevance=0.6, bloom_level_min=4,
        dimension_affinity=("consistency", "logical_rigor"),
    ),
    RelationType.COREFERENCE: RelationSpec(
        type=RelationType.COREFERENCE,
        branch=RelationBranch.EQUIVALENCE,
        label_zh="同指关系",
        label_en="Coreference",
        is_symmetric=True,
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("consistency",),
    ),

    # ── 2. Hierarchy ──────────────────────────────────────────────────────────
    RelationType.HYPERNYM: RelationSpec(
        type=RelationType.HYPERNYM,
        branch=RelationBranch.HIERARCHY,
        label_zh="上位词（过宽泛化）",
        label_en="Hypernym",
        is_transitive=True,
        product_relevance=0.8, bloom_level_min=3,
        dimension_affinity=("user_focus", "problem_clarity"),
        wordnet_relation="hypernym", conceptnet_relation="IsA",
    ),
    RelationType.HYPONYM: RelationSpec(
        type=RelationType.HYPONYM,
        branch=RelationBranch.HIERARCHY,
        label_zh="下位词（具体化）",
        label_en="Hyponym",
        is_transitive=True,
        product_relevance=0.8, bloom_level_min=3,
        dimension_affinity=("user_focus", "problem_clarity", "actionability"),
        wordnet_relation="hyponym",
    ),
    RelationType.INSTANCE_OF: RelationSpec(
        type=RelationType.INSTANCE_OF,
        branch=RelationBranch.HIERARCHY,
        label_zh="实例关系",
        label_en="Instance Of",
        product_relevance=0.6, bloom_level_min=3,
        dimension_affinity=("evidence_quality",),
        wordnet_relation="instance_hypernym",
    ),
    RelationType.PART_OF: RelationSpec(
        type=RelationType.PART_OF,
        branch=RelationBranch.HIERARCHY,
        label_zh="整体-部分",
        label_en="Part Of",
        is_transitive=True,
        product_relevance=0.7, bloom_level_min=3,
        dimension_affinity=("completeness", "actionability"),
        wordnet_relation="part_meronym", conceptnet_relation="PartOf",
    ),
    RelationType.MEMBER_OF: RelationSpec(
        type=RelationType.MEMBER_OF,
        branch=RelationBranch.HIERARCHY,
        label_zh="集合-成员",
        label_en="Member Of",
        product_relevance=0.5, bloom_level_min=3,
        dimension_affinity=("user_focus",),
        wordnet_relation="member_meronym",
    ),

    # ── 3. Approximation ─────────────────────────────────────────────────────
    RelationType.NEAR_SYNONYM: RelationSpec(
        type=RelationType.NEAR_SYNONYM,
        branch=RelationBranch.APPROXIMATION,
        label_zh="近义关系",
        label_en="Near Synonym",
        is_symmetric=True,
        product_relevance=0.5, bloom_level_min=5,
        dimension_affinity=("consistency",),
        wordnet_relation="similar",
    ),
    RelationType.ANALOGY: RelationSpec(
        type=RelationType.ANALOGY,
        branch=RelationBranch.APPROXIMATION,
        label_zh="类比关系",
        label_en="Analogy",
        product_relevance=0.6, bloom_level_min=5,
        dimension_affinity=("depth_of_insight", "competitive"),
    ),
    RelationType.METAPHOR: RelationSpec(
        type=RelationType.METAPHOR,
        branch=RelationBranch.APPROXIMATION,
        label_zh="隐喻关系",
        label_en="Metaphor",
        product_relevance=0.4, bloom_level_min=5,
        dimension_affinity=("depth_of_insight",),
    ),
    RelationType.POLYSEMY: RelationSpec(
        type=RelationType.POLYSEMY,
        branch=RelationBranch.APPROXIMATION,
        label_zh="多义词关联",
        label_en="Polysemy",
        is_symmetric=True,
        product_relevance=0.3, bloom_level_min=4,
        dimension_affinity=("consistency",),
    ),

    # ── 4. Semantic Distance ──────────────────────────────────────────────────
    RelationType.SAME_FIELD: RelationSpec(
        type=RelationType.SAME_FIELD,
        branch=RelationBranch.SEMANTIC_DISTANCE,
        label_zh="同场关系",
        label_en="Same Field",
        is_symmetric=True,
        product_relevance=0.4, bloom_level_min=2,
        dimension_affinity=("scenario_relevance",),
    ),
    RelationType.FRAME_BASED: RelationSpec(
        type=RelationType.FRAME_BASED,
        branch=RelationBranch.SEMANTIC_DISTANCE,
        label_zh="同框架关系",
        label_en="Frame-based",
        product_relevance=0.5, bloom_level_min=3,
        dimension_affinity=("completeness",),
    ),
    RelationType.COLLOCATION: RelationSpec(
        type=RelationType.COLLOCATION,
        branch=RelationBranch.SEMANTIC_DISTANCE,
        label_zh="搭配邻近",
        label_en="Collocation",
        is_symmetric=True,
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("structure",),
    ),
    RelationType.THEMATIC: RelationSpec(
        type=RelationType.THEMATIC,
        branch=RelationBranch.SEMANTIC_DISTANCE,
        label_zh="主题关联",
        label_en="Thematic",
        product_relevance=0.4, bloom_level_min=2,
        dimension_affinity=("scenario_relevance",),
    ),

    # ── 5. Opposition ─────────────────────────────────────────────────────────
    RelationType.COMPLEMENTARY_ANTONYM: RelationSpec(
        type=RelationType.COMPLEMENTARY_ANTONYM,
        branch=RelationBranch.OPPOSITION,
        label_zh="互补反义",
        label_en="Complementary Antonym",
        is_symmetric=True,
        product_relevance=0.7, bloom_level_min=4,
        dimension_affinity=("strategic_thinking", "competitive"),
        wordnet_relation="antonym", conceptnet_relation="Antonym",
    ),
    RelationType.GRADABLE_ANTONYM: RelationSpec(
        type=RelationType.GRADABLE_ANTONYM,
        branch=RelationBranch.OPPOSITION,
        label_zh="程度反义",
        label_en="Gradable Antonym",
        is_symmetric=True,
        product_relevance=0.5, bloom_level_min=4,
        dimension_affinity=("depth_of_insight",),
    ),
    RelationType.CONVERSE: RelationSpec(
        type=RelationType.CONVERSE,
        branch=RelationBranch.OPPOSITION,
        label_zh="关系反义（角色互换）",
        label_en="Converse",
        is_symmetric=True,
        product_relevance=0.7, bloom_level_min=4,
        dimension_affinity=("strategic_thinking", "completeness"),
    ),
    RelationType.DIRECTIONAL_ANTONYM: RelationSpec(
        type=RelationType.DIRECTIONAL_ANTONYM,
        branch=RelationBranch.OPPOSITION,
        label_zh="方向反义",
        label_en="Directional Antonym",
        is_symmetric=True,
        product_relevance=0.4, bloom_level_min=3,
        dimension_affinity=("strategic_thinking",),
    ),
    RelationType.CONTRADICTION: RelationSpec(
        type=RelationType.CONTRADICTION,
        branch=RelationBranch.OPPOSITION,
        label_zh="矛盾关系",
        label_en="Contradiction",
        is_symmetric=True,
        product_relevance=0.9, bloom_level_min=4,
        dimension_affinity=("consistency", "logical_rigor"),
    ),

    # ── 6. Temporal / Sequential ──────────────────────────────────────────────
    RelationType.TEMPORAL_SEQUENCE: RelationSpec(
        type=RelationType.TEMPORAL_SEQUENCE,
        branch=RelationBranch.TEMPORAL,
        label_zh="时序关系",
        label_en="Temporal Sequence",
        is_transitive=True,
        product_relevance=0.7, bloom_level_min=3,
        dimension_affinity=("actionability",),
    ),
    RelationType.CAUSE_EFFECT: RelationSpec(
        type=RelationType.CAUSE_EFFECT,
        branch=RelationBranch.TEMPORAL,
        label_zh="因果关系",
        label_en="Cause-Effect",
        product_relevance=0.95, bloom_level_min=4,
        dimension_affinity=("logical_rigor", "evidence_quality", "depth_of_insight"),
        conceptnet_relation="Causes",
    ),
    RelationType.CONDITION_CONSEQUENCE: RelationSpec(
        type=RelationType.CONDITION_CONSEQUENCE,
        branch=RelationBranch.TEMPORAL,
        label_zh="条件-结果",
        label_en="Condition-Consequence",
        product_relevance=0.85, bloom_level_min=4,
        dimension_affinity=("strategic_thinking", "depth_of_insight"),
    ),
    RelationType.PURPOSE_MEANS: RelationSpec(
        type=RelationType.PURPOSE_MEANS,
        branch=RelationBranch.TEMPORAL,
        label_zh="目的-手段",
        label_en="Purpose-Means",
        product_relevance=0.8, bloom_level_min=3,
        dimension_affinity=("actionability", "metrics"),
    ),
    RelationType.CAUSATIVE: RelationSpec(
        type=RelationType.CAUSATIVE,
        branch=RelationBranch.TEMPORAL,
        label_zh="使役关系",
        label_en="Causative",
        product_relevance=0.6, bloom_level_min=4,
        dimension_affinity=("logical_rigor",),
    ),
    RelationType.PROCESS_STAGE: RelationSpec(
        type=RelationType.PROCESS_STAGE,
        branch=RelationBranch.TEMPORAL,
        label_zh="过程阶段",
        label_en="Process Stage",
        product_relevance=0.7, bloom_level_min=3,
        dimension_affinity=("actionability", "completeness"),
    ),

    # ── 7. Functional ─────────────────────────────────────────────────────────
    RelationType.TOOL_PURPOSE: RelationSpec(
        type=RelationType.TOOL_PURPOSE,
        branch=RelationBranch.FUNCTIONAL,
        label_zh="工具-目的",
        label_en="Tool-Purpose",
        product_relevance=0.7, bloom_level_min=3,
        dimension_affinity=("actionability", "channel_feasibility"),
        conceptnet_relation="UsedFor",
    ),
    RelationType.AGENT_ACTION: RelationSpec(
        type=RelationType.AGENT_ACTION,
        branch=RelationBranch.FUNCTIONAL,
        label_zh="施事-行为",
        label_en="Agent-Action",
        product_relevance=0.6, bloom_level_min=3,
        dimension_affinity=("actionability", "completeness"),
    ),
    RelationType.ACTION_RESULT: RelationSpec(
        type=RelationType.ACTION_RESULT,
        branch=RelationBranch.FUNCTIONAL,
        label_zh="行为-结果",
        label_en="Action-Result",
        product_relevance=0.7, bloom_level_min=3,
        dimension_affinity=("metrics", "evidence_quality"),
    ),
    RelationType.ACTION_PATIENT: RelationSpec(
        type=RelationType.ACTION_PATIENT,
        branch=RelationBranch.FUNCTIONAL,
        label_zh="行为-受事",
        label_en="Action-Patient",
        product_relevance=0.5, bloom_level_min=3,
        dimension_affinity=("user_focus",),
    ),
    RelationType.FUNCTIONAL_EQUIVALENCE: RelationSpec(
        type=RelationType.FUNCTIONAL_EQUIVALENCE,
        branch=RelationBranch.FUNCTIONAL,
        label_zh="功能等价",
        label_en="Functional Equivalence",
        is_symmetric=True,
        product_relevance=0.6, bloom_level_min=4,
        dimension_affinity=("competitive",),
    ),

    # ── 8. Evaluative ─────────────────────────────────────────────────────────
    RelationType.POSITIVE_CONNOTATION: RelationSpec(
        type=RelationType.POSITIVE_CONNOTATION,
        branch=RelationBranch.EVALUATIVE,
        label_zh="褒义表述",
        label_en="Positive Connotation",
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("evidence_quality",),
    ),
    RelationType.NEGATIVE_CONNOTATION: RelationSpec(
        type=RelationType.NEGATIVE_CONNOTATION,
        branch=RelationBranch.EVALUATIVE,
        label_zh="贬义表述",
        label_en="Negative Connotation",
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("evidence_quality",),
    ),
    RelationType.EUPHEMISM: RelationSpec(
        type=RelationType.EUPHEMISM,
        branch=RelationBranch.EVALUATIVE,
        label_zh="委婉表述",
        label_en="Euphemism",
        product_relevance=0.3, bloom_level_min=2,
        dimension_affinity=("depth_of_insight",),
    ),
    RelationType.VAGUE_QUANTIFIER: RelationSpec(
        type=RelationType.VAGUE_QUANTIFIER,
        branch=RelationBranch.EVALUATIVE,
        label_zh="模糊量化",
        label_en="Vague Quantifier",
        product_relevance=0.8, bloom_level_min=3,
        dimension_affinity=("evidence_quality", "metrics"),
    ),
    # ── 9. Evidence ──────────────────────────────────────────────────────────
    RelationType.EVIDENCE_CITATION: RelationSpec(
        type=RelationType.EVIDENCE_CITATION,
        branch=RelationBranch.TEMPORAL,
        label_zh="证据引用",
        label_en="Evidence Citation",
        product_relevance=0.9, bloom_level_min=4,
        dimension_affinity=("evidence_quality", "metrics"),
    ),
}


# ---------------------------------------------------------------------------
# Helper lookups
# ---------------------------------------------------------------------------

def get_branch_relations(branch: RelationBranch) -> list[RelationSpec]:
    return [s for s in RELATION_ONTOLOGY.values() if s.branch == branch]


def get_high_relevance_relations(threshold: float = 0.7) -> list[RelationSpec]:
    return [s for s in RELATION_ONTOLOGY.values() if s.product_relevance >= threshold]


def get_relations_for_dimension(dimension: str) -> list[RelationSpec]:
    return [
        s for s in RELATION_ONTOLOGY.values()
        if dimension in s.dimension_affinity
    ]


def get_relations_for_bloom(bloom_level: int) -> list[RelationSpec]:
    """Relations whose minimum Bloom level is at most *bloom_level*."""
    return [s for s in RELATION_ONTOLOGY.values() if s.bloom_level_min <= bloom_level]


BRANCH_DISPLAY: dict[RelationBranch, dict[str, str]] = {
    RelationBranch.EQUIVALENCE: {"zh": "等阶关系", "en": "Equivalence"},
    RelationBranch.HIERARCHY: {"zh": "从属关系", "en": "Hierarchy"},
    RelationBranch.APPROXIMATION: {"zh": "近似关系", "en": "Approximation"},
    RelationBranch.SEMANTIC_DISTANCE: {"zh": "语义远近", "en": "Semantic Distance"},
    RelationBranch.OPPOSITION: {"zh": "对立关系", "en": "Opposition"},
    RelationBranch.TEMPORAL: {"zh": "时序/因果", "en": "Temporal / Causal"},
    RelationBranch.FUNCTIONAL: {"zh": "功能关系", "en": "Functional"},
    RelationBranch.EVALUATIVE: {"zh": "评价/情感", "en": "Evaluative"},
}
