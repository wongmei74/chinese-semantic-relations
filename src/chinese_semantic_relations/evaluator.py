"""
Semantic Response Evaluator — scores text based on semantic relation
richness, correctness, and compositional depth.

Scoring components:
  1. Relation diversity — how many different relation types are used
  2. Branch coverage — how many of the 8 branches are represented
  3. High-relevance hit rate — proportion of high-value relations present
  4. Composition depth — presence of relation type combinations
  5. Anti-patterns — penalties for vague quantifiers, unsupported claims

Zero LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ontology import (
    RelationBranch,
    RelationType,
    RELATION_ONTOLOGY,
    get_high_relevance_relations,
)
from .analyzer import RelationProfile


@dataclass
class SemanticScore:
    """Semantic quality score for a user response or conversation."""

    relation_diversity_score: float = 0.0
    branch_coverage_score: float = 0.0
    high_relevance_score: float = 0.0
    composition_score: float = 0.0
    anti_pattern_penalty: float = 0.0
    total: float = 0.0
    max_possible: float = 25.0

    breakdown: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


# Relation compositions that signal expert-level thinking
_EXPERT_COMPOSITIONS: list[tuple[str, set[RelationType], float]] = [
    (
        "因果+证据或量化",
        {RelationType.CAUSE_EFFECT, RelationType.EVIDENCE_CITATION},
        2.0,
    ),
    (
        "因果+具体化",
        {RelationType.CAUSE_EFFECT, RelationType.HYPONYM},
        1.5,
    ),
    (
        "层级+对立",
        {RelationType.HYPONYM, RelationType.COMPLEMENTARY_ANTONYM},
        2.5,
    ),
    (
        "时序+条件",
        {RelationType.TEMPORAL_SEQUENCE, RelationType.CONDITION_CONSEQUENCE},
        2.0,
    ),
    (
        "目的+手段+结果",
        {RelationType.PURPOSE_MEANS, RelationType.ACTION_RESULT},
        2.0,
    ),
    (
        "整体分解+阶段",
        {RelationType.PART_OF, RelationType.PROCESS_STAGE},
        1.5,
    ),
    (
        "竞品+反义",
        {RelationType.FUNCTIONAL_EQUIVALENCE, RelationType.COMPLEMENTARY_ANTONYM},
        2.0,
    ),
    (
        "双边+施事",
        {RelationType.CONVERSE, RelationType.AGENT_ACTION},
        1.5,
    ),
    (
        "因果+条件+对立",
        {RelationType.CAUSE_EFFECT, RelationType.CONDITION_CONSEQUENCE, RelationType.COMPLEMENTARY_ANTONYM},
        3.0,
    ),
]

# Anti-patterns: relation types whose presence should reduce score
_ANTI_PATTERNS: dict[RelationType, tuple[float, str]] = {
    RelationType.VAGUE_QUANTIFIER: (
        2.0,
        "回答中存在模糊量化——建议用具体数据替代'很多'、'大部分'等词",
    ),
    RelationType.EUPHEMISM: (
        1.0,
        "委婉表述可能掩盖了真实问题——建议更直接地描述挑战和风险",
    ),
    RelationType.POSITIVE_CONNOTATION: (
        0.5,
        "主观褒义词过多——建议减少营销语言，增加客观论证",
    ),
}

# Branch importance weights for product training
_BRANCH_WEIGHTS: dict[RelationBranch, float] = {
    RelationBranch.TEMPORAL: 1.0,
    RelationBranch.HIERARCHY: 0.9,
    RelationBranch.OPPOSITION: 0.85,
    RelationBranch.FUNCTIONAL: 0.8,
    RelationBranch.EQUIVALENCE: 0.5,
    RelationBranch.APPROXIMATION: 0.5,
    RelationBranch.SEMANTIC_DISTANCE: 0.3,
    RelationBranch.EVALUATIVE: 0.2,
}


class SemanticResponseEvaluator:
    """Evaluates semantic quality of user responses.

    Produces a SemanticScore on a 0-25 scale (compatible with DualScorer
    dimension scales).
    """

    def evaluate(
        self,
        profile: RelationProfile,
        *,
        max_score: float = 25.0,
    ) -> SemanticScore:

        diversity = self._score_diversity(profile)
        branch_cov = self._score_branch_coverage(profile)
        relevance = self._score_high_relevance(profile)
        composition = self._score_composition(profile)
        penalty = self._compute_penalty(profile)

        raw = diversity + branch_cov + relevance + composition - penalty
        total = max(0.0, min(max_score, raw))

        recommendations = self._generate_recommendations(profile)

        return SemanticScore(
            relation_diversity_score=round(diversity, 2),
            branch_coverage_score=round(branch_cov, 2),
            high_relevance_score=round(relevance, 2),
            composition_score=round(composition, 2),
            anti_pattern_penalty=round(penalty, 2),
            total=round(total, 2),
            max_possible=max_score,
            breakdown={
                "detected_types": len(profile.detected),
                "branch_count": profile.branch_diversity,
                "branches_present": [b.value for b in profile.detected_branches],
                "compositions_found": self._list_compositions(profile),
                "anti_patterns": [
                    rt.value for rt in _ANTI_PATTERNS
                    if profile.has(rt)
                ],
            },
            recommendations=recommendations,
        )

    def _score_diversity(self, profile: RelationProfile) -> float:
        """0-7 points: diversity of relation types used."""
        count = profile.relation_diversity
        if count == 0:
            return 0.0
        return min(7.0, count * 0.8)

    def _score_branch_coverage(self, profile: RelationProfile) -> float:
        """0-6 points: weighted branch coverage."""
        total = 0.0
        for branch, weight in _BRANCH_WEIGHTS.items():
            if profile.branch_strength(branch) > 0:
                total += weight
        return min(6.0, total)

    def _score_high_relevance(self, profile: RelationProfile) -> float:
        """0-6 points: presence of high product-relevance relation types."""
        high_rel = get_high_relevance_relations(threshold=0.7)
        hits = sum(1 for spec in high_rel if profile.has(spec.type))
        ratio = hits / max(len(high_rel), 1)
        return min(6.0, ratio * 6.0)

    def _score_composition(self, profile: RelationProfile) -> float:
        """0-6 points: presence of expert-level relation compositions."""
        total = 0.0
        for _, required_types, bonus in _EXPERT_COMPOSITIONS:
            if required_types.issubset(profile.detected_types):
                total += bonus
        return min(6.0, total)

    def _compute_penalty(self, profile: RelationProfile) -> float:
        """Penalty for anti-patterns (capped at 4.0)."""
        total = 0.0
        has_evidence = profile.has(RelationType.EVIDENCE_CITATION)
        for rtype, (penalty_val, _) in _ANTI_PATTERNS.items():
            det = profile.detected.get(rtype)
            if det and det.match_count >= 2:
                p = penalty_val
                if rtype == RelationType.POSITIVE_CONNOTATION and has_evidence:
                    p *= 0.35
                total += p * min(det.match_count / 3, 1.5)
        return min(4.0, total)

    def _list_compositions(self, profile: RelationProfile) -> list[str]:
        found: list[str] = []
        for name, required_types, _ in _EXPERT_COMPOSITIONS:
            if required_types.issubset(profile.detected_types):
                found.append(name)
        return found

    def _generate_recommendations(self, profile: RelationProfile) -> list[str]:
        recs: list[str] = []

        critical_missing = {
            RelationType.CAUSE_EFFECT: "缺少因果推理——请论证'为什么'，而不只是'是什么'",
            RelationType.CONDITION_CONSEQUENCE: "缺少条件分析——你的方案建立在什么假设上？如果假设不成立呢？",
            RelationType.COMPLEMENTARY_ANTONYM: "缺少对立面分析——竞品怎么看？最坏情况是什么？",
            RelationType.HYPONYM: "论述过于笼统——请聚焦到具体的用户群/场景/数据",
            RelationType.TEMPORAL_SEQUENCE: "缺少时间规划——按什么顺序执行？里程碑是什么？",
        }

        for rtype, msg in critical_missing.items():
            if not profile.has(rtype):
                recs.append(msg)

        for rtype, (_, msg) in _ANTI_PATTERNS.items():
            det = profile.detected.get(rtype)
            if det and det.match_count >= 2:
                recs.append(msg)

        return recs[:5]
