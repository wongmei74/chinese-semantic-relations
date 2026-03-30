"""
Semantic Challenge Strategy Mapper.

Maps semantic relation gaps to targeted challenge strategies.  Each gap
type generates a specific probe instruction that can be appended to any
AI challenge/interview prompt.

Three levels of probing intensity:
  L1 引导 — surface relations, scaffolding questions
  L2 正常 — intermediate relations, specific demands
  L3 高压 — deep compositional relations, zero-tolerance precision

Usage::

    from chinese_semantic_relations import SemanticChallengeStrategyMapper, SemanticRelationAnalyzer

    analyzer = SemanticRelationAnalyzer()
    mapper = SemanticChallengeStrategyMapper()

    profile = analyzer.analyze("我们的AI平台帮助所有学生提高成绩")
    plan = mapper.plan_from_profile(profile, difficulty_level=2)
    print(plan.instruction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .ontology import RelationBranch, RelationSpec, RelationType, RELATION_ONTOLOGY
from .analyzer import RelationProfile


@dataclass
class SemanticProbe:
    """A targeted question/instruction derived from a semantic gap."""

    relation_type: RelationType
    branch: RelationBranch
    gap_description_zh: str
    probe_instruction: str
    dimension_affinity: list[str] = field(default_factory=list)
    bloom_level: int = 3
    product_relevance: float = 0.5
    recommended_personas: list[str] = field(default_factory=list)


@dataclass
class SemanticStrategyPlan:
    """Strategy plan with semantic relation metadata."""

    strategy_key: str
    instruction: str
    probes: list[SemanticProbe] = field(default_factory=list)
    dimension_affinity: list[str] = field(default_factory=list)
    confidence: float = 0.5
    relation_context: dict = field(default_factory=dict)
    fallback_instruction: str = "请基于用户回答进行追问或反驳。"


# Type for pluggable persona resolver
PersonaResolver = Callable[[tuple[str, ...]], list[str]]


def _default_persona_resolver(dims: tuple[str, ...]) -> list[str]:
    """Default no-op: returns empty list. Override to map dimensions to personas."""
    return []


# ---------------------------------------------------------------------------
# Probe templates per relation type × difficulty level
# ---------------------------------------------------------------------------

_PROBE_TEMPLATES: dict[RelationType, dict[int, str]] = {
    RelationType.CAUSE_EFFECT: {
        1: (
            "你提到了一个因果关系，能展开说说吗？"
            "具体是什么原因导致了这个结果？中间有哪些关键环节？"
        ),
        2: (
            "你说 X 导致 Y，但你有证据排除反向因果吗？"
            "会不会是 Y 先存在才产生了 X？或者有第三个因素同时影响了两者？"
        ),
        3: (
            "你的因果链存在跳跃——从 A 到 B 之间至少缺了 2 个中间变量。"
            "请列出完整的因果传导路径，并为每个环节提供可量化的转化率。"
            "如果任一环节断裂，你的替代方案是什么？"
        ),
    },
    RelationType.CONDITION_CONSEQUENCE: {
        1: "你的方案看起来建立在一些假设上，能明确列出这些前提条件吗？",
        2: (
            "你假设了 X 条件成立，但如果这个条件不满足会怎样？"
            "你有 Plan B 吗？触发 Plan B 的判断标准是什么？"
        ),
        3: (
            "你的方案依赖 3 个隐含前提同时成立（概率估算？）。"
            "请为每个前提提供验证方式和失败时的降级策略，"
            "并计算最坏情况下的损失规模。"
        ),
    },
    RelationType.HYPERNYM: {
        1: "你提到了一个比较宽泛的概念，能具体化一下吗？聚焦到最核心的那一类。",
        2: (
            "你说'所有用户都需要这个'——真的是所有用户吗？"
            "哪类用户最迫切？优先级怎么排？用什么标准判定？"
        ),
        3: (
            "'所有用户'是一个危险信号。请用 MECE 原则把用户拆成 3-5 个互斥细分，"
            "每个细分附带规模、痛点强度、付费意愿，然后告诉我 Day 1 只服务哪一个。"
        ),
    },
    RelationType.HYPONYM: {
        1: "你聚焦在一个很具体的场景，它能代表更大的市场吗？",
        2: (
            "你的方案针对非常细分的用户群——这个群体有多大？"
            "如何从这个切入点扩展到相邻市场？扩展路径是什么？"
        ),
        3: (
            "你的 TAM 被限制在一个极小的细分里。"
            "请论证：(1) 这个细分足以支撑独立商业模式；"
            "(2) 如果不能，你的扩展路线图（每阶段新增的细分及其规模）。"
        ),
    },
    RelationType.COMPLEMENTARY_ANTONYM: {
        1: "你的方案有没有考虑过'反面'的情况？如果事情朝相反方向发展呢？",
        2: (
            "你提到了 X 的优势，但它的对立面是什么？"
            "竞品在这个维度上是什么策略？你的差异化壁垒是什么？"
        ),
        3: (
            "你的方案假设了一个单一正面走向。请构建一个'最坏情况'矩阵：\n"
            "- 竞品在 6 个月内复制你的核心功能\n"
            "- 市场需求下降 50%\n"
            "- 关键合作伙伴终止合作\n"
            "每种情况你的应对策略和预估损失是什么？"
        ),
    },
    RelationType.CONTRADICTION: {
        1: "你的描述里有一些地方似乎前后不太一致，能再检查一下吗？",
        2: (
            "你前面说了 X，但后面又说了与 X 矛盾的 Y。"
            "这两个论点不能同时成立——你选择保留哪个？为什么？"
        ),
        3: (
            "你的论证中存在结构性矛盾：\n"
            "论点 A（你在第 N 轮提出的）与论点 B（你刚才说的）逻辑互斥。"
            "请明确选择一个方向，并废弃另一个，同时说明废弃方向的机会成本。"
        ),
    },
    RelationType.PURPOSE_MEANS: {
        1: "你说了要实现什么目标，能具体说说打算用什么方法来实现吗？",
        2: (
            "你的目标和手段之间有一段'跳跃'。"
            "从手段 X 到目标 Y，中间的转化逻辑是什么？效率预期是多少？"
        ),
        3: (
            "你有明确目标但缺乏可执行的手段链。请拆解成：\n"
            "目标 → 关键举措（≤3个）→ 每个举措的资源需求 → 预期产出 → 风险点。"
            "不接受'然后就自然实现了'这种跳跃。"
        ),
    },
    RelationType.TEMPORAL_SEQUENCE: {
        1: "你的计划按什么顺序推进？先做什么？后做什么？",
        2: (
            "你列出了要做的事情，但没有给出时间线。"
            "Q1、Q2 分别完成什么？关键里程碑和判定标准是什么？"
        ),
        3: (
            "你的路线图缺乏关键路径分析。请识别：\n"
            "- 哪些任务存在前置依赖（不能并行）\n"
            "- 关键路径上最大的延期风险是什么\n"
            "- 如果关键路径延期 2 周，对整体交付的影响是什么"
        ),
    },
    RelationType.PROCESS_STAGE: {
        1: "你的流程有几个环节？能简单画一下从用户进入到最终完成的路径吗？",
        2: (
            "你的流程描述不够完整——用户在第 2 步和第 4 步之间发生了什么？"
            "每个环节的转化率预期是多少？"
        ),
        3: (
            "请提供完整的用户旅程漏斗：\n"
            "每阶段的定义、转化率预期、关键脱落原因、以及针对脱落的挽回策略。"
            "如果漏斗某一层转化率低于预期 50%，你的止损方案是什么？"
        ),
    },
    RelationType.VAGUE_QUANTIFIER: {
        1: "你说了'很多用户'——能给个大概的数字范围吗？",
        2: (
            "你用了模糊量词（'大部分'、'很多'、'显著'）。"
            "请把每一个模糊量词替换成具体数字。数据来源是什么？"
        ),
        3: (
            "你的回答中有 N 处模糊量化，在严肃的商业讨论中这是不可接受的。"
            "请逐一替换为：精确数字 + 数据来源 + 置信区间。"
            "如果你没有数据，请明确说'我目前没有数据，需要验证'。"
        ),
    },
    RelationType.TOOL_PURPOSE: {
        1: "你提到了一些工具或方法，能说说具体用来解决什么问题吗？",
        2: (
            "你选择了工具 X 来实现目标 Y——为什么是 X 而不是备选方案？"
            "X 的局限性是什么？什么情况下需要换用其他方案？"
        ),
        3: (
            "请为每个关键工具/技术选型提供：\n"
            "(1) 选择理由和 2 个备选方案\n"
            "(2) 每个方案的优劣对比矩阵\n"
            "(3) 切换成本和锁定风险评估"
        ),
    },
    RelationType.FUNCTIONAL_EQUIVALENCE: {
        1: "市场上有没有类似功能的产品？你和它们的区别是什么？",
        2: (
            "你说你的产品有独特价值，但竞品 Z 似乎提供了功能等价的方案。"
            "用户为什么选你而不是 Z？这个差异可持续吗？"
        ),
        3: (
            "请构建一个竞争矩阵：你 vs 3 个功能等价竞品，\n"
            "对比维度包括：核心功能、价格、用户体验、切换成本、护城河。\n"
            "在每个维度上量化你的优势或劣势百分比。"
        ),
    },
    RelationType.PART_OF: {
        1: "你的方案涉及几个关键模块？它们之间是什么关系？",
        2: (
            "你描述了整体方案但没有拆解。"
            "核心系统由哪些子模块组成？哪个子模块是 MVP 必须的？哪些可以延后？"
        ),
        3: (
            "请提供完整的系统分解：\n"
            "- L1 系统组成（3-5 个核心模块）\n"
            "- 每个模块的输入/输出接口\n"
            "- 模块间的依赖关系图\n"
            "- 独立可交付的最小子集（MVP Scope）"
        ),
    },
    RelationType.ANALOGY: {
        1: "能用一个类比来帮助理解你的产品吗？比如'X 界的 Y'？",
        2: (
            "你类比了 X，但这个类比在哪些方面成立？哪些方面不成立？"
            "类比的局限性是什么？"
        ),
        3: (
            "你的类比（'我们是 X 行业的 Y'）表面成立但深层有断裂：\n"
            "Y 的成功依赖于 Z 条件，但在你的行业里 Z 不存在。"
            "请给出一个更精确的类比，或者停止用类比来回避真正的商业逻辑论证。"
        ),
    },
    RelationType.CONVERSE: {
        1: "你谈到了供给侧，需求侧呢？两边是如何匹配的？",
        2: (
            "你关注了关系中的一方，但另一方的视角呢？"
            "买方和卖方/供方和需方/教方和学方——另一方的需求和动机是什么？"
        ),
        3: (
            "你的双边模型中，冷启动的鸡蛋问题怎么解决？"
            "请分别描述供给侧和需求侧的启动策略、获客成本、以及达到网络效应的临界点。"
        ),
    },
    RelationType.POSITIVE_CONNOTATION: {
        1: "你用了一些很积极的词汇来描述产品，能用具体数据来支撑吗？",
        2: (
            "你说产品'颠覆性'、'领先'、'创新'——这些是主观评价还是有客观标准？"
            "用什么指标来衡量'颠覆'？和行业基准相比领先多少？"
        ),
        3: (
            "你的描述中充满了营销语言而非论证。请移除所有主观褒义词，"
            "改用可量化的事实陈述。例如：'颠覆性' → '比现有方案效率提升 X%'。"
        ),
    },
    RelationType.AGENT_ACTION: {
        1: "这件事谁来做？团队里谁负责？",
        2: (
            "你描述了很多行动，但主体不明确。"
            "每个关键行动的负责人是谁？他们有相应的能力和资源吗？"
        ),
        3: (
            "请建立一个 RACI 矩阵：\n"
            "列出 5 个关键任务 × 相关角色，明确谁是 Responsible、Accountable、"
            "Consulted、Informed。目前你的团队能力是否覆盖所有 R 角色？"
        ),
    },
}

# Extra relation expectations by sector
_SECTOR_EXPECTED_EXTRA: dict[str, set[RelationType]] = {
    "education": {
        RelationType.CONVERSE,
        RelationType.INSTANCE_OF,
        RelationType.PART_OF,
    },
    "saas": {
        RelationType.FUNCTIONAL_EQUIVALENCE,
        RelationType.TOOL_PURPOSE,
        RelationType.CONDITION_CONSEQUENCE,
    },
    "fintech": {
        RelationType.CONDITION_CONSEQUENCE,
        RelationType.CONTRADICTION,
        RelationType.PART_OF,
    },
    "healthcare": {
        RelationType.CONDITION_CONSEQUENCE,
        RelationType.CONVERSE,
        RelationType.PART_OF,
    },
}


_DEFAULT_PROBES: dict[int, str] = {
    1: "能对这个观点做进一步展开吗？",
    2: "请提供更具体的论据来支撑这个观点。",
    3: "这个论点缺乏足够的支撑，请提供数据、案例和逻辑链。",
}

_GAP_DESCRIPTIONS: dict[RelationType, str] = {
    RelationType.CAUSE_EFFECT: "因果推理缺失——主张了结果但未论证原因",
    RelationType.CONDITION_CONSEQUENCE: "条件推理缺失——方案缺少前提假设分析",
    RelationType.HYPERNYM: "定义过于宽泛——未将概念具体化到可操作级别",
    RelationType.HYPONYM: "缺乏具体化——概念停留在抽象层面",
    RelationType.COMPLEMENTARY_ANTONYM: "对立面分析缺失——未考虑反面情况或竞争威胁",
    RelationType.CONTRADICTION: "存在前后矛盾——论证链中出现逻辑互斥",
    RelationType.PURPOSE_MEANS: "目的-手段断裂——目标明确但缺乏实现路径",
    RelationType.TEMPORAL_SEQUENCE: "时序规划缺失——缺少执行顺序和里程碑",
    RelationType.PROCESS_STAGE: "流程不完整——缺少关键阶段或转化环节",
    RelationType.VAGUE_QUANTIFIER: "量化不足——使用了模糊量词替代精确数据",
    RelationType.TOOL_PURPOSE: "工具-目的不匹配——选型理由不充分",
    RelationType.FUNCTIONAL_EQUIVALENCE: "竞品对标缺失——未与功能等价方案做对比",
    RelationType.PART_OF: "结构分解缺失——未将整体拆解为可管理的组件",
    RelationType.ANALOGY: "类比可能失真——类比的适用边界未明确",
    RelationType.CONVERSE: "双边视角缺失——只关注了关系的一方",
    RelationType.POSITIVE_CONNOTATION: "主观评价过多——褒义描述缺乏客观支撑",
    RelationType.AGENT_ACTION: "责任主体模糊——谁做什么不明确",
}


class SemanticChallengeStrategyMapper:
    """Maps semantic relation gaps to challenge strategy plans.

    Args:
        persona_resolver: Optional callable that maps dimension affinity
            tuples to a list of recommended persona keys. If not provided,
            recommended_personas will be empty. This allows downstream
            systems (e.g. AI training platforms) to plug in their own
            persona/role mappings.
    """

    def __init__(
        self,
        *,
        persona_resolver: PersonaResolver | None = None,
    ) -> None:
        self._persona_resolver = persona_resolver or _default_persona_resolver

    def plan_from_profile(
        self,
        profile: RelationProfile,
        *,
        difficulty_level: int = 2,
        target_dimension: str | None = None,
        expected_types: set[RelationType] | None = None,
        sector: str | None = None,
        max_probes: int = 3,
    ) -> SemanticStrategyPlan:
        """Generate a strategy plan from a RelationProfile.

        Args:
            profile: Output from SemanticRelationAnalyzer.analyze()
            difficulty_level: 1=引导, 2=正常, 3=高压
            target_dimension: If set, prioritize gaps related to this dimension
            expected_types: Relation types expected for this context
            sector: Optional sector key (education, saas, fintech, healthcare)
            max_probes: Maximum number of probes to generate
        """
        sector_key = sector.lower().strip() if sector else ""
        baseline = _default_expected_types(difficulty_level)
        if sector_key:
            baseline |= _SECTOR_EXPECTED_EXTRA.get(sector_key, set())
        if expected_types is None:
            expected_types = baseline
        else:
            expected_types = set(expected_types) | baseline

        gaps = self._rank_gaps(
            profile, expected_types,
            target_dimension=target_dimension,
        )

        if not gaps:
            weak = self._find_weak_relations(profile, difficulty_level)
            if weak:
                gaps = weak

        probes: list[SemanticProbe] = []
        for rtype, relevance in gaps[:max_probes]:
            spec = RELATION_ONTOLOGY.get(rtype)
            templates = _PROBE_TEMPLATES.get(rtype, _DEFAULT_PROBES)
            level = min(difficulty_level, max(templates.keys()))
            instruction = templates.get(level, templates.get(2, _DEFAULT_PROBES[2]))

            dims = tuple(spec.dimension_affinity) if spec else ()
            probes.append(SemanticProbe(
                relation_type=rtype,
                branch=spec.branch if spec else RelationBranch.TEMPORAL,
                gap_description_zh=_GAP_DESCRIPTIONS.get(rtype, "语义关系缺失"),
                probe_instruction=instruction,
                dimension_affinity=list(dims),
                bloom_level=spec.bloom_level_min if spec else 3,
                product_relevance=relevance,
                recommended_personas=self._persona_resolver(dims),
            ))

        if not probes:
            return SemanticStrategyPlan(
                strategy_key="semantic_default",
                instruction=_DEFAULT_PROBES.get(difficulty_level, _DEFAULT_PROBES[2]),
                confidence=0.2,
                fallback_instruction=_DEFAULT_PROBES[2],
            )

        primary = probes[0]
        all_dims: list[str] = []
        for p in probes:
            all_dims.extend(d for d in p.dimension_affinity if d not in all_dims)

        return SemanticStrategyPlan(
            strategy_key=f"semantic_{primary.relation_type.value}",
            instruction=primary.probe_instruction,
            probes=probes,
            dimension_affinity=all_dims,
            confidence=min(0.9, 0.4 + primary.product_relevance * 0.5),
            relation_context={
                "primary_gap": primary.relation_type.value,
                "gap_count": len(gaps),
                "profile_diversity": profile.relation_diversity,
                "branch_diversity": profile.branch_diversity,
                "difficulty_level": difficulty_level,
                "sector": sector_key or None,
            },
        )

    def _rank_gaps(
        self,
        profile: RelationProfile,
        expected: set[RelationType],
        *,
        target_dimension: str | None = None,
    ) -> list[tuple[RelationType, float]]:
        missing = expected - profile.detected_types
        if not missing:
            return []

        ranked: list[tuple[RelationType, float]] = []
        for rtype in missing:
            spec = RELATION_ONTOLOGY.get(rtype)
            if not spec:
                continue
            score = spec.product_relevance

            if target_dimension and target_dimension in spec.dimension_affinity:
                score *= 1.5

            ranked.append((rtype, min(score, 1.0)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _find_weak_relations(
        self,
        profile: RelationProfile,
        difficulty_level: int,
    ) -> list[tuple[RelationType, float]]:
        weak: list[tuple[RelationType, float]] = []
        for rtype, det in profile.detected.items():
            spec = RELATION_ONTOLOGY.get(rtype)
            if not spec or spec.bloom_level_min > difficulty_level + 2:
                continue
            rel = spec.product_relevance
            if det.strength < 0.5:
                weak.append((rtype, rel))
            elif det.quality < 0.42 and rel >= 0.55:
                weak.append((rtype, rel * 0.85))

        weak.sort(key=lambda x: x[1], reverse=True)
        return weak


def _default_expected_types(difficulty_level: int) -> set[RelationType]:
    """Baseline set of expected relation types, expanding with difficulty."""
    base = {
        RelationType.CAUSE_EFFECT,
        RelationType.PURPOSE_MEANS,
        RelationType.HYPONYM,
    }
    if difficulty_level >= 2:
        base |= {
            RelationType.CONDITION_CONSEQUENCE,
            RelationType.COMPLEMENTARY_ANTONYM,
            RelationType.PROCESS_STAGE,
            RelationType.TEMPORAL_SEQUENCE,
        }
    if difficulty_level >= 3:
        base |= {
            RelationType.CONTRADICTION,
            RelationType.CONVERSE,
            RelationType.PART_OF,
            RelationType.FUNCTIONAL_EQUIVALENCE,
        }
    return base
