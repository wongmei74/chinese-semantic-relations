"""Tests for Chinese Semantic Relations (zero-LLM, regex-based).

Covers all four sub-modules:
  1. ontology   — type registry consistency
  2. analyzer   — relation detection from text
  3. strategies — gap → challenge strategy mapping
  4. evaluator  — semantic quality scoring
"""

import pytest

from chinese_semantic_relations import (
    SemanticRelationEngine,
    SemanticRelationAnalyzer,
    SemanticChallengeStrategyMapper,
    SemanticResponseEvaluator,
    RelationProfile,
    ConversationSemanticTrajectory,
    SemanticStrategyPlan,
    SemanticScore,
    StrategyPlan,
    RelationType,
    RelationBranch,
    RELATION_ONTOLOGY,
    BRANCH_DISPLAY,
    get_branch_relations,
    get_high_relevance_relations,
    get_relations_for_dimension,
    get_relations_for_bloom,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> SemanticRelationEngine:
    return SemanticRelationEngine()


@pytest.fixture
def analyzer() -> SemanticRelationAnalyzer:
    return SemanticRelationAnalyzer()


@pytest.fixture
def mapper() -> SemanticChallengeStrategyMapper:
    return SemanticChallengeStrategyMapper()


@pytest.fixture
def evaluator() -> SemanticResponseEvaluator:
    return SemanticResponseEvaluator()


# =========================================================================
# 1. Ontology Tests
# =========================================================================

class TestOntology:
    def test_all_relation_types_have_specs(self) -> None:
        for rtype in RelationType:
            assert rtype in RELATION_ONTOLOGY, f"Missing spec for {rtype}"

    def test_all_branches_have_display(self) -> None:
        for branch in RelationBranch:
            assert branch in BRANCH_DISPLAY

    def test_high_relevance_returns_subset(self) -> None:
        high = get_high_relevance_relations(0.7)
        assert len(high) > 0
        assert all(s.product_relevance >= 0.7 for s in high)

    def test_branch_relations_complete(self) -> None:
        total = sum(len(get_branch_relations(b)) for b in RelationBranch)
        assert total == len(RELATION_ONTOLOGY)

    def test_dimension_affinity_lookup(self) -> None:
        logical = get_relations_for_dimension("logical_rigor")
        assert len(logical) > 0
        assert all("logical_rigor" in s.dimension_affinity for s in logical)

    def test_bloom_level_filter(self) -> None:
        basic = get_relations_for_bloom(2)
        advanced = get_relations_for_bloom(5)
        assert len(basic) <= len(advanced)

    def test_relation_count(self) -> None:
        assert len(RELATION_ONTOLOGY) == 38


# =========================================================================
# 2. Analyzer Tests
# =========================================================================

class TestAnalyzerCausal:
    @pytest.mark.parametrize("text", [
        "因为需求不清晰所以延期了",
        "The delay happened because requirements were unclear.",
        "由于市场变化导致策略调整",
    ])
    def test_cause_effect_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.CAUSE_EFFECT)

    @pytest.mark.parametrize("text", [
        "如果预算充足我们可以扩展团队",
        "Unless we secure funding, the project will stall.",
        "一旦获得融资就可以加速",
    ])
    def test_condition_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.CONDITION_CONSEQUENCE)


class TestAnalyzerHierarchy:
    @pytest.mark.parametrize("text", [
        "所有用户都需要这个功能",
        "Every customer expects fast delivery.",
        "任何人都可以使用",
    ])
    def test_hypernym_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.HYPERNYM)

    @pytest.mark.parametrize("text", [
        "我们产品的所有功能都很稳定",
        "All features ship in Q2.",
        "每个功能都要过回归测试",
    ])
    def test_hypernym_not_triggered_on_all_features_scope(
        self, analyzer: SemanticRelationAnalyzer, text: str
    ) -> None:
        profile = analyzer.analyze(text)
        assert not profile.has(RelationType.HYPERNYM)

    @pytest.mark.parametrize("text", [
        "具体来说我们聚焦在B端SaaS",
        "For example, we target small restaurants in Tier-2 cities.",
        "尤其是25-35岁的女性用户",
    ])
    def test_hyponym_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.HYPONYM)

    @pytest.mark.parametrize("text", [
        "系统包含三个核心模块：用户管理、订单处理和支付",
        "The platform consists of a frontend, backend, and database.",
        "由推荐引擎、搜索和内容管理三部分组成",
    ])
    def test_part_of_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.PART_OF)


class TestAnalyzerOpposition:
    @pytest.mark.parametrize("text", [
        "我们的方案与竞品不同于在精准度上",
        "Rather than competing on price, we differentiate on quality.",
        "与传统方案截然不同的是",
    ])
    def test_complementary_antonym_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.COMPLEMENTARY_ANTONYM)

    @pytest.mark.parametrize("text", [
        "一方面要控制成本另一方面又要提升质量",
        "The plan contradicts what we said earlier about budget constraints.",
        "既要增长又要盈利这是矛盾的",
    ])
    def test_contradiction_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.CONTRADICTION)


class TestAnalyzerTemporal:
    @pytest.mark.parametrize("text", [
        "首先完成MVP然后进行用户测试接下来优化迭代",
        "In Q1 we build the core, then launch beta in Q2.",
        "第一步调研第二步设计第三步开发",
    ])
    def test_temporal_sequence_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.TEMPORAL_SEQUENCE)

    @pytest.mark.parametrize("text", [
        "为了提高用户留存率我们计划通过个性化推荐来实现",
        "We leverage AI in order to reduce manual processing time.",
        "目的是降低获客成本通过社交裂变来实现",
    ])
    def test_purpose_means_detected(self, analyzer: SemanticRelationAnalyzer, text: str) -> None:
        profile = analyzer.analyze(text)
        assert profile.has(RelationType.PURPOSE_MEANS)


class TestAnalyzerProfile:
    def test_empty_text_returns_empty_profile(self, analyzer: SemanticRelationAnalyzer) -> None:
        profile = analyzer.analyze("")
        assert profile.relation_diversity == 0
        assert profile.branch_diversity == 0

    def test_rich_text_high_diversity(self, analyzer: SemanticRelationAnalyzer) -> None:
        text = (
            "因为市场需求旺盛所以我们决定扩展。"
            "如果融资顺利就能加速。"
            "首先做MVP然后测试。"
            "与竞品不同的是我们更快。"
            "具体来说聚焦25-35岁用户。"
            "系统包含前端和后端两个模块。"
            "用户可以上传文件和查看报告。"
            "为了提高留存通过推荐实现。"
        )
        profile = analyzer.analyze(text)
        assert profile.relation_diversity >= 6
        assert profile.branch_diversity >= 4

    def test_conversation_analysis(self, analyzer: SemanticRelationAnalyzer) -> None:
        rounds = [
            {"user_response": "因为用户需求大所以我们决定做这个产品"},
            {"user_response": "具体来说我们聚焦在教育行业"},
            {"user_response": ""},
        ]
        traj = analyzer.analyze_conversation(rounds)
        assert isinstance(traj, ConversationSemanticTrajectory)
        assert len(traj.per_round) == 3
        assert traj.merged.has(RelationType.CAUSE_EFFECT)
        assert traj.merged.has(RelationType.HYPONYM)
        assert "new_types_per_round" in traj.growth


class TestAnalyzerCompare:
    def test_compare_finds_missing_branches(self, analyzer: SemanticRelationAnalyzer) -> None:
        profile = analyzer.analyze("因为市场大所以值得做")
        gaps = analyzer.compare(profile)
        assert "missing_types" in gaps
        assert len(gaps["missing_types"]) > 0


# =========================================================================
# 3. Strategy Tests
# =========================================================================

class TestStrategyMapper:
    def test_missing_causal_generates_probe(self, mapper: SemanticChallengeStrategyMapper) -> None:
        profile = RelationProfile()
        plan = mapper.plan_from_profile(profile, difficulty_level=2)
        assert plan.probes
        assert plan.confidence > 0.3

    def test_difficulty_affects_instruction(self, mapper: SemanticChallengeStrategyMapper) -> None:
        profile = RelationProfile()
        plan_l1 = mapper.plan_from_profile(profile, difficulty_level=1)
        plan_l3 = mapper.plan_from_profile(profile, difficulty_level=3)
        assert plan_l1.instruction != plan_l3.instruction

    def test_target_dimension_prioritizes_gaps(self, mapper: SemanticChallengeStrategyMapper) -> None:
        profile = RelationProfile()
        plan = mapper.plan_from_profile(
            profile, difficulty_level=2, target_dimension="evidence_quality",
        )
        assert plan.probes
        assert any("evidence_quality" in p.dimension_affinity for p in plan.probes)

    def test_sector_merges_into_context(self, mapper: SemanticChallengeStrategyMapper) -> None:
        profile = RelationProfile()
        plan = mapper.plan_from_profile(profile, difficulty_level=2, sector="saas")
        assert plan.relation_context.get("sector") == "saas"

    def test_custom_persona_resolver(self) -> None:
        resolver = lambda dims: ["custom_role"] if "metrics" in dims else []
        mapper = SemanticChallengeStrategyMapper(persona_resolver=resolver)
        profile = RelationProfile()
        plan = mapper.plan_from_profile(
            profile, difficulty_level=2, target_dimension="metrics",
        )
        assert plan.probes
        found = any("custom_role" in p.recommended_personas for p in plan.probes)
        assert found


# =========================================================================
# 4. Evaluator Tests
# =========================================================================

class TestEvaluator:
    def test_empty_profile_zero_score(self, evaluator: SemanticResponseEvaluator) -> None:
        profile = RelationProfile()
        score = evaluator.evaluate(profile)
        assert score.total == 0.0

    def test_rich_profile_high_score(
        self, analyzer: SemanticRelationAnalyzer, evaluator: SemanticResponseEvaluator
    ) -> None:
        text = (
            "因为市场需求旺盛所以我们决定扩展产品线。"
            "如果Q1融资成功就能在Q2推出MVP。"
            "首先完成用户调研然后设计原型接下来开发测试。"
            "与竞品不同的是我们用AI实现了个性化推荐。"
            "具体来说我们聚焦25-35岁的城市白领，"
            "系统包含推荐引擎、内容管理和支付三个核心模块。"
            "通过社交裂变来降低获客成本。"
        )
        profile = analyzer.analyze(text)
        score = evaluator.evaluate(profile)
        assert score.total > 10.0

    def test_vague_text_gets_penalty(
        self, analyzer: SemanticRelationAnalyzer, evaluator: SemanticResponseEvaluator
    ) -> None:
        text = "很多用户都觉得大部分功能非常显著地提升了效率，有些人特别喜欢"
        profile = analyzer.analyze(text)
        score = evaluator.evaluate(profile)
        assert score.anti_pattern_penalty > 0

    def test_composition_cause_plus_evidence(
        self, analyzer: SemanticRelationAnalyzer, evaluator: SemanticResponseEvaluator
    ) -> None:
        text = "因为调研显示留存提升了30%，所以我们加大投放。"
        profile = analyzer.analyze(text)
        score = evaluator.evaluate(profile)
        assert profile.has(RelationType.CAUSE_EFFECT)
        assert profile.has(RelationType.EVIDENCE_CITATION)
        assert "因果+证据或量化" in score.breakdown.get("compositions_found", [])


# =========================================================================
# 5. Engine Facade Tests
# =========================================================================

class TestEngineFacade:
    def test_plan_challenge(self, engine: SemanticRelationEngine) -> None:
        plan = engine.plan_challenge(
            "我们的AI平台帮助所有学生提高成绩",
            difficulty_level=2,
        )
        assert isinstance(plan, SemanticStrategyPlan)
        assert len(plan.instruction) > 0

    def test_evaluate_response(self, engine: SemanticRelationEngine) -> None:
        score = engine.evaluate_response(
            "因为我们通过个性化推荐算法帮助具体用户提升了30%学习效率"
        )
        assert isinstance(score, SemanticScore)
        assert score.total >= 0

    def test_analyze(self, engine: SemanticRelationEngine) -> None:
        profile = engine.analyze("因为X所以Y")
        assert isinstance(profile, RelationProfile)
        assert profile.has(RelationType.CAUSE_EFFECT)

    def test_to_strategy_plan(self, engine: SemanticRelationEngine) -> None:
        plan = engine.plan_challenge("我们做一个AI产品")
        base = engine.to_strategy_plan(plan)
        assert isinstance(base, StrategyPlan)
        assert base.strategy_key == plan.strategy_key

    def test_health_check(self, engine: SemanticRelationEngine) -> None:
        hc = engine.health_check()
        assert hc["ok"] is True
        assert hc["relation_types"] == 38

    def test_full_pipeline(self, engine: SemanticRelationEngine) -> None:
        idea = "我们做一个在线教育平台，帮助学生提高成绩。"
        plan = engine.plan_challenge(idea, difficulty_level=2, sector="education")
        assert plan.probes
        assert plan.confidence > 0.3

        response = (
            "因为K12赛道在线渗透率从15%提升到45%，"
            "具体来说聚焦三四线城市初中生数学。"
            "如果Q1获得1000种子用户通过口碑裂变Q2达到5000人。"
            "与现有产品不同的是我们用知识图谱实现个性化出题。"
        )
        score = engine.evaluate_response(response)
        assert score.total > 8.0
