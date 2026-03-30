"""
Semantic Relation Analyzer — zero-LLM text analysis.

Extracts semantic relations from Chinese/English text using regex-based
feature detection.  Produces a ``RelationProfile`` summarizing which
relation types are present, their strength, and textual evidence.

Pipeline: text → detect → RelationProfile (detected relations + gaps)

Feeds into strategies.py (gap → challenge) and evaluator.py (scoring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ontology import RelationBranch, RelationType


@dataclass
class DetectedRelation:
    """A single semantic relation detected in text."""

    relation_type: RelationType
    branch: RelationBranch
    evidence: list[str] = field(default_factory=list)
    match_count: int = 0
    strength: float = 0.0  # 0-1, computed from match density
    quality: float = 0.5   # 0-1, relation quality (evidence backing)


@dataclass
class RelationProfile:
    """Aggregated semantic relation profile of a text."""

    detected: dict[RelationType, DetectedRelation] = field(default_factory=dict)
    branch_coverage: dict[RelationBranch, float] = field(default_factory=dict)
    total_strength: float = 0.0
    relation_diversity: int = 0
    branch_diversity: int = 0
    text_length: int = 0
    average_quality: float = 0.0

    @property
    def detected_types(self) -> set[RelationType]:
        return set(self.detected.keys())

    @property
    def detected_branches(self) -> set[RelationBranch]:
        return {d.branch for d in self.detected.values()}

    def has(self, relation_type: RelationType) -> bool:
        return relation_type in self.detected

    def strength_of(self, relation_type: RelationType) -> float:
        det = self.detected.get(relation_type)
        return det.strength if det else 0.0

    def branch_strength(self, branch: RelationBranch) -> float:
        return self.branch_coverage.get(branch, 0.0)


@dataclass
class ConversationSemanticTrajectory:
    """Multi-round semantic analysis preserving per-round temporal info."""

    per_round: list[RelationProfile]
    merged: RelationProfile
    growth: dict  # {"new_types_per_round": [...], "diversity_trend": [...]}
    cross_round_contradictions: list[dict]


# ---------------------------------------------------------------------------
# Regex detector registry — bilingual (Chinese + English)
# ---------------------------------------------------------------------------

_RELATION_PATTERNS: dict[RelationType, list[str]] = {
    # ── Equivalence ───────────────────────────────────────────────────────────
    RelationType.STRICT_SYNONYM: [
        r"(等于|等同于|就是|即是|也就是|其实就是|本质上是)",
        r"(is\s+equivalent\s+to|same\s+as|identical\s+to)",
    ],
    RelationType.LOGICAL_EQUIVALENCE: [
        r"(定义为|是指|意思是|指的是|换言之|也可以说)",
        r"(is\s+defined\s+as|refers\s+to|in\s+other\s+words|means\s+that)",
    ],
    RelationType.COREFERENCE: [
        r"(也叫|又称|即|也就是说|简称)",
        r"(also\s+known\s+as|a\.k\.a|i\.e\.|namely)",
    ],
    RelationType.TERMINOLOGICAL_SYNONYM: [
        r"(术语|专业术语|行业术语|技术名词|学名)",
        r"(technical\s+term|jargon|terminology|formally\s+known\s+as)",
    ],

    # ── Hierarchy ─────────────────────────────────────────────────────────────
    RelationType.HYPERNYM: [
        r"(所有|全部|每个|任何|大家都|一切|所有人|所有用户|所有客户)",
        r"(all|every|any|everyone|everything|universal)",
        r"(整体上|宏观来看|总体而言|广泛地)",
    ],
    RelationType.HYPONYM: [
        r"(具体来说|特别是|尤其是|比如说|以.{0,15}为例|细分为|分为)",
        r"(specifically|in\s+particular|especially|for\s+example|such\s+as|e\.g\.)",
        r"(其中.{0,8}(一种|一类|一个|一项))",
    ],
    RelationType.INSTANCE_OF: [
        r"(案例|实例|真实场景|实际案例|用户故事)",
        r"(case\s+study|real[\s-]world\s+example|instance|use\s+case)",
    ],
    RelationType.PART_OF: [
        r"(组成|包含|包括|由.{0,15}(组成|构成)|分为.{0,15}(部分|模块|环节))",
        r"(consists?\s+of|comprises?|includes?|is\s+composed\s+of|made\s+up\s+of)",
        r"(其中|一部分|子系统|子模块|核心模块|环节)",
    ],
    RelationType.MEMBER_OF: [
        r"(属于|归入|隶属|归类|是.{0,8}(成员|部分|一员))",
        r"(belongs?\s+to|member\s+of|part\s+of\s+the\s+group)",
    ],

    # ── Approximation ─────────────────────────────────────────────────────────
    RelationType.NEAR_SYNONYM: [
        r"(类似于|类似|差不多|基本上是|相当于|近似)",
        r"(similar\s+to|akin\s+to|comparable\s+to|roughly|approximately)",
    ],
    RelationType.ANALOGY: [
        r"(就像|好比|相当于|如同|类似于.{0,15}一样|好像)",
        r"(like\s+(?:a|an|the)\b|as\s+if|analogous\s+to|think\s+of\s+it\s+as)",
        r"(可以类比|对标|对标.{0,15}的)",
    ],
    RelationType.METAPHOR: [
        r"(比喻|可以看作|是.{0,10}的.{0,10}(引擎|大脑|心脏|基石|桥梁|催化剂))",
        r"(metaphor|figuratively|is\s+the\s+(\w+\s+){0,2}(engine|brain|heart|backbone))",
    ],
    RelationType.POLYSEMY: [
        r"(这里的.{0,10}是指|此处.{0,10}含义|在.{0,10}语境下)",
        r"(in\s+this\s+context|here\s+.{0,15}means|sense\s+of\s+the\s+word)",
    ],

    # ── Semantic Distance ─────────────────────────────────────────────────────
    RelationType.SAME_FIELD: [
        r"(同(领域|行业|赛道|类型|品类)|相关(行业|领域))",
        r"(same\s+(field|industry|sector|category)|related\s+(industry|domain))",
    ],
    RelationType.FRAME_BASED: [
        r"(生态(系统|圈)|产业链|价值链|供应链|上下游)",
        r"(ecosystem|value\s+chain|supply\s+chain|upstream|downstream)",
    ],
    RelationType.THEMATIC: [
        r"(在.{0,15}(领域|行业|赛道|市场|场景)中?)",
        r"(in\s+the\s+(\w+\s+){0,2}(industry|sector|market|domain|space))",
    ],
    RelationType.COLLOCATION: [
        r"(深度整合|紧密配合|无缝衔接|协同效应|强关联)",
        r"(tightly\s+integrated|work\s+in\s+tandem|synergy\s+between|closely\s+coupled)",
    ],

    # ── Opposition ────────────────────────────────────────────────────────────
    RelationType.COMPLEMENTARY_ANTONYM: [
        r"(而不是|而非|不同于|与.{0,15}相反|截然不同|相反地|反之)",
        r"(rather\s+than|instead\s+of|unlike|contrary\s+to|as\s+opposed\s+to|on\s+the\s+contrary)",
    ],
    RelationType.GRADABLE_ANTONYM: [
        r"(更(多|少|高|低|快|慢|强|弱|好|差|大|小|重|轻))",
        r"(more|less|higher|lower|faster|slower|stronger|weaker|better|worse)",
        r"(比.{0,20}(更|还要))",
    ],
    RelationType.CONVERSE: [
        r"(买.{0,8}卖|供.{0,8}需|教.{0,8}学|雇.{0,8}被雇|投资.{0,8}回报)",
        r"(buy.{0,15}sell|supply.{0,15}demand|teach.{0,15}learn|give.{0,15}receive)",
        r"(双边|双方|互为|互相|相互|对等)",
    ],
    RelationType.CONTRADICTION: [
        r"(矛盾|自相矛盾|不一致|冲突|前后不一|逻辑矛盾)",
        r"(contradict|inconsisten|conflict|paradox|mutually\s+exclusive)",
        r"(一方面.{5,80}另一方面|既要.{5,40}又要)",
    ],

    # ── Temporal / Sequential ─────────────────────────────────────────────────
    RelationType.TEMPORAL_SEQUENCE: [
        r"(首先|然后|接下来|最后|第[一二三四五六七八九十\d]步|之后|随后|紧接着)",
        r"(first|then|next|finally|step\s+\d|after\s+that|subsequently|followed\s+by)",
        r"(V[12345]|阶段[一二三四五]|Phase\s*[1-5]|Q[1-4]|第[一二三四]季度)",
    ],
    RelationType.CAUSE_EFFECT: [
        r"(因为|由于|导致|所以|因此|造成|引起|带来|根本原因)",
        r"(because|since|due\s+to|therefore|thus|hence|results?\s+in|leads?\s+to|causes?)",
        r"(原因是|根源在于|归因于|这是因为|之所以)",
    ],
    RelationType.CONDITION_CONSEQUENCE: [
        r"(如果|假设|假如|当.{0,40}时|除非|一旦|万一|前提是)",
        r"(if|assuming|given\s+that|unless|once|in\s+case|provided\s+that|suppose)",
        r"(只有.{0,30}才|不.{0,15}就|否则)",
    ],
    RelationType.PURPOSE_MEANS: [
        r"(为了|目的是|旨在|用来|通过.{0,30}(实现|达到|完成)|借助|利用.{0,30}来)",
        r"(in\s+order\s+to|so\s+that|aim\s+to|to\s+achieve|by\s+means\s+of|leverage.{0,20}to)",
        r"(目标是|最终目标|核心目标)",
    ],
    RelationType.CAUSATIVE: [
        r"(使得|让.{0,15}(变得|成为|能够)|促使|推动|驱动|赋能)",
        r"(enable|empower|drive|make\s+\w+\s+(possible|easier|better)|facilitate)",
    ],
    RelationType.PROCESS_STAGE: [
        r"(流程|步骤|阶段|环节|工序|管线|链路|漏斗|Pipeline)",
        r"(workflow|pipeline|funnel|process|procedure|stage\s+\d|phase\s+\d)",
        r"(从.{0,20}到.{0,20}(的过程|的流程|的链路))",
    ],

    # ── Functional ────────────────────────────────────────────────────────────
    RelationType.TOOL_PURPOSE: [
        r"(用于|用来|可以用.{0,10}来|作为.{0,15}(工具|手段|方法|渠道))",
        r"(used\s+for|serves?\s+as|acts?\s+as\s+a\s+tool|functions?\s+as)",
    ],
    RelationType.AGENT_ACTION: [
        r"(用户.{0,15}(会|可以|需要|应该)|客户.{0,15}(使用|购买|选择)|团队.{0,15}负责)",
        r"(the\s+user\s+(will|can|should)|is\s+responsible\s+for|team\s+\w+\s+(handles?|manages?))",
        r"(由.{0,10}(负责|执行|运营|管理|开发))",
        r"(\w+\s+is\s+responsible\s+for|\w+\s+(handles?|manages?|leads?|owns?)\s+)",
    ],
    RelationType.ACTION_RESULT: [
        r"(产出|输出|产生|生成|带来.{0,15}(结果|效果|影响|产出))",
        r"(results?\s+in|produces?|generates?|yields?|outcomes?)",
        r"(效果是|结果是|最终效果|成果)",
    ],
    RelationType.FUNCTIONAL_EQUIVALENCE: [
        r"(替代|可以替代|功能等价|同类产品|可替换|替代品)",
        r"(alternative\s+(to|solutions?)|substitute|replacement|equivalent\s+functionality)",
        r"(竞品|对标产品|同类型)",
    ],
    RelationType.ACTION_PATIENT: [
        r"(服务于|面向.{0,12}(用户|客户|群体)|针对.{0,12}(用户|客户|群体|市场|场景))",
        r"(serves?\s+(\w+\s+){0,3}(users?|customers?)|targeting\s+(\w+\s+){0,3}(users?|market))",
    ],

    # ── Evaluative ────────────────────────────────────────────────────────────
    RelationType.VAGUE_QUANTIFIER: [
        r"(很多|大部分|大量|显著|不少|相当|普遍|常见|有些|一些)",
        r"(many|most|a\s+lot|significant|some|several|frequently|substantial)",
        r"(一定程度上|某种程度|在很大程度上)",
    ],
    RelationType.POSITIVE_CONNOTATION: [
        r"(领先|颠覆|革命性|独特|强大|最好|卓越|突破|创新|极致)",
        r"(leading|disruptive|revolutionary|unique|best|excellent|breakthrough|innovative)",
    ],
    RelationType.NEGATIVE_CONNOTATION: [
        r"(落后|过时|劣势|缺陷|风险|痛点|障碍|瓶颈|困境)",
        r"(outdated|disadvantage|flaw|risk|bottleneck|challenge|obstacle|pain\s+point)",
    ],
    RelationType.EUPHEMISM: [
        r"(有待(提升|改善|优化)|空间较大|仍需努力|面临挑战)",
        r"(room\s+for\s+improvement|opportunity\s+area|evolving)",
    ],

    # ── Evidence ─────────────────────────────────────────────────────────────
    RelationType.EVIDENCE_CITATION: [
        r"(\d+\.?\d*)\s*(%|万|亿|元|美元|USD|倍|[xX]|次|天|月|年|人|家|个)",
        r"(根据|来自|引用|数据显示|报告|调研|研究表明|统计)",
        r"(according\s+to|based\s+on|data\s+shows|research\s+indicates|survey|study)",
        r"(例如|比如|以.{0,30}为例|案例|like|such\s+as|for\s+example|case\s+study)",
    ],
}

# Pre-compiled patterns for performance
_COMPILED_PATTERNS: dict[RelationType, list[re.Pattern[str]]] | None = None


def _get_compiled() -> dict[RelationType, list[re.Pattern[str]]]:
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = {
            rtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rtype, patterns in _RELATION_PATTERNS.items()
        }
    return _COMPILED_PATTERNS


# Narrow false positives: universal quantifier scoped to *features/modules*, not
# "所有用户都需要这个功能" (users are the scope; 功能 is object of need).
_HYPERNYM_FALSE_POSITIVE_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"所有功能|全部功能|每个功能|任何功能|一切功能"),
    re.compile(r"所有模块|全部模块|每个模块"),
    re.compile(r"所有特性|全部环节|每个步骤"),
    re.compile(r"all\s+features?|every\s+feature|all\s+modules?|all\s+capabilities", re.IGNORECASE),
]


# Map each RelationType to its branch (for fast lookup)
_TYPE_BRANCH: dict[RelationType, RelationBranch] = {
    # Equivalence
    RelationType.STRICT_SYNONYM: RelationBranch.EQUIVALENCE,
    RelationType.LOGICAL_EQUIVALENCE: RelationBranch.EQUIVALENCE,
    RelationType.COREFERENCE: RelationBranch.EQUIVALENCE,
    RelationType.TERMINOLOGICAL_SYNONYM: RelationBranch.EQUIVALENCE,
    # Hierarchy
    RelationType.HYPERNYM: RelationBranch.HIERARCHY,
    RelationType.HYPONYM: RelationBranch.HIERARCHY,
    RelationType.INSTANCE_OF: RelationBranch.HIERARCHY,
    RelationType.PART_OF: RelationBranch.HIERARCHY,
    RelationType.MEMBER_OF: RelationBranch.HIERARCHY,
    # Approximation
    RelationType.NEAR_SYNONYM: RelationBranch.APPROXIMATION,
    RelationType.ANALOGY: RelationBranch.APPROXIMATION,
    RelationType.METAPHOR: RelationBranch.APPROXIMATION,
    RelationType.POLYSEMY: RelationBranch.APPROXIMATION,
    # Semantic Distance
    RelationType.THEMATIC: RelationBranch.SEMANTIC_DISTANCE,
    RelationType.SAME_FIELD: RelationBranch.SEMANTIC_DISTANCE,
    RelationType.FRAME_BASED: RelationBranch.SEMANTIC_DISTANCE,
    RelationType.COLLOCATION: RelationBranch.SEMANTIC_DISTANCE,
    # Opposition
    RelationType.COMPLEMENTARY_ANTONYM: RelationBranch.OPPOSITION,
    RelationType.GRADABLE_ANTONYM: RelationBranch.OPPOSITION,
    RelationType.CONVERSE: RelationBranch.OPPOSITION,
    RelationType.CONTRADICTION: RelationBranch.OPPOSITION,
    # Temporal
    RelationType.TEMPORAL_SEQUENCE: RelationBranch.TEMPORAL,
    RelationType.CAUSE_EFFECT: RelationBranch.TEMPORAL,
    RelationType.CONDITION_CONSEQUENCE: RelationBranch.TEMPORAL,
    RelationType.PURPOSE_MEANS: RelationBranch.TEMPORAL,
    RelationType.CAUSATIVE: RelationBranch.TEMPORAL,
    RelationType.PROCESS_STAGE: RelationBranch.TEMPORAL,
    # Functional
    RelationType.TOOL_PURPOSE: RelationBranch.FUNCTIONAL,
    RelationType.AGENT_ACTION: RelationBranch.FUNCTIONAL,
    RelationType.ACTION_RESULT: RelationBranch.FUNCTIONAL,
    RelationType.FUNCTIONAL_EQUIVALENCE: RelationBranch.FUNCTIONAL,
    RelationType.ACTION_PATIENT: RelationBranch.FUNCTIONAL,
    # Evaluative
    RelationType.VAGUE_QUANTIFIER: RelationBranch.EVALUATIVE,
    RelationType.POSITIVE_CONNOTATION: RelationBranch.EVALUATIVE,
    RelationType.NEGATIVE_CONNOTATION: RelationBranch.EVALUATIVE,
    RelationType.EUPHEMISM: RelationBranch.EVALUATIVE,
    # Evidence
    RelationType.EVIDENCE_CITATION: RelationBranch.TEMPORAL,
}


# Quantitative evidence detector (reused from dual_scorer pattern)
_QUANT_EVIDENCE_RE = re.compile(
    r"(\d+\.?\d*)\s*(%|万|亿|元|美元|USD|倍|[xX]|次|天|月|年|人|家|个)",
    re.IGNORECASE,
)


class SemanticRelationAnalyzer:
    """Zero-LLM semantic relation extractor.

    Scans text with compiled regex patterns, counts matches, computes
    per-relation strength, and assembles a ``RelationProfile``.
    """

    EVIDENCE_SNIPPET_RADIUS: int = 30

    def analyze(self, text: str) -> RelationProfile:
        if not text or not text.strip():
            return RelationProfile()

        compiled = _get_compiled()
        detected: dict[RelationType, DetectedRelation] = {}
        text_len = len(text)

        for rtype, patterns in compiled.items():
            total_matches = 0
            evidence_snippets: list[str] = []

            for pat in patterns:
                for m in pat.finditer(text):
                    total_matches += 1
                    if len(evidence_snippets) < 3:
                        start = max(0, m.start() - self.EVIDENCE_SNIPPET_RADIUS)
                        end = min(text_len, m.end() + self.EVIDENCE_SNIPPET_RADIUS)
                        evidence_snippets.append(text[start:end].strip())

            if total_matches > 0:
                density = min(total_matches / max(text_len / 200, 1), 1.0)
                strength = min(1.0, 0.3 + density * 0.7)

                detected[rtype] = DetectedRelation(
                    relation_type=rtype,
                    branch=_TYPE_BRANCH[rtype],
                    evidence=evidence_snippets,
                    match_count=total_matches,
                    strength=strength,
                )

        # ── HYPERNYM: drop matches that sit inside "all features / all modules" phrasing
        if RelationType.HYPERNYM in detected:
            hypernym_patterns = compiled[RelationType.HYPERNYM]
            filtered_count = 0
            filtered_evidence: list[str] = []
            for pat in hypernym_patterns:
                for m in pat.finditer(text):
                    win_s = max(0, m.start() - 12)
                    win_e = min(text_len, m.end() + 24)
                    window = text[win_s:win_e]
                    excluded = any(fp.search(window) for fp in _HYPERNYM_FALSE_POSITIVE_PHRASES)
                    if not excluded:
                        filtered_count += 1
                        if len(filtered_evidence) < 3:
                            snip_s = max(0, m.start() - self.EVIDENCE_SNIPPET_RADIUS)
                            snip_e = min(text_len, m.end() + self.EVIDENCE_SNIPPET_RADIUS)
                            filtered_evidence.append(text[snip_s:snip_e].strip())
            if filtered_count == 0:
                del detected[RelationType.HYPERNYM]
            else:
                det = detected[RelationType.HYPERNYM]
                det.match_count = filtered_count
                det.evidence = filtered_evidence
                density = min(filtered_count / max(text_len / 200, 1), 1.0)
                det.strength = min(1.0, 0.3 + density * 0.7)

        has_quant = bool(_QUANT_EVIDENCE_RE.search(text))

        # ── Fix 1: Compute relation quality via co-occurrence ────────────────
        for rtype, det in detected.items():
            if rtype in (RelationType.CAUSE_EFFECT, RelationType.CONDITION_CONSEQUENCE) and has_quant:
                det.quality = 0.8
            elif rtype == RelationType.CAUSE_EFFECT and RelationType.VAGUE_QUANTIFIER in detected:
                det.quality = 0.3
            elif rtype == RelationType.HYPERNYM and RelationType.HYPONYM in detected:
                det.quality = 0.7
            elif rtype == RelationType.POSITIVE_CONNOTATION:
                det.quality = 0.7 if has_quant else 0.3

        branch_coverage: dict[RelationBranch, float] = {}
        for branch in RelationBranch:
            branch_relations = [d for d in detected.values() if d.branch == branch]
            if branch_relations:
                branch_coverage[branch] = sum(d.strength for d in branch_relations) / len(branch_relations)
            else:
                branch_coverage[branch] = 0.0

        avg_quality = (
            sum(d.quality for d in detected.values()) / len(detected)
            if detected else 0.0
        )

        return RelationProfile(
            detected=detected,
            branch_coverage=branch_coverage,
            total_strength=sum(d.strength for d in detected.values()),
            relation_diversity=len(detected),
            branch_diversity=sum(1 for v in branch_coverage.values() if v > 0),
            text_length=text_len,
            average_quality=avg_quality,
        )

    def _merge_conversation_text(self, rounds: list[dict]) -> RelationProfile:
        """Merge all user responses into a single text and analyze."""
        combined = ""
        for r in rounds:
            resp = r.get("user_response", "")
            if resp:
                combined += resp + "\n"
        return self.analyze(combined)

    def analyze_conversation(
        self, rounds: list[dict]
    ) -> ConversationSemanticTrajectory:
        """Analyze each round individually, track growth, and detect contradictions."""
        per_round: list[RelationProfile] = []
        seen_types: set[RelationType] = set()
        new_types_per_round: list[list[str]] = []
        diversity_trend: list[int] = []

        for r in rounds:
            resp = r.get("user_response", "")
            profile = self.analyze(resp) if resp else RelationProfile()
            per_round.append(profile)

            current_types = profile.detected_types
            new_types = current_types - seen_types
            new_types_per_round.append([t.value for t in new_types])
            seen_types |= current_types
            diversity_trend.append(profile.relation_diversity)

        contradictions: list[dict] = []
        for i, pi in enumerate(per_round):
            for j in range(i + 1, len(per_round)):
                pj = per_round[j]
                if pi.has(RelationType.HYPERNYM) and pj.has(RelationType.HYPONYM):
                    contradictions.append({
                        "type": "generalization_narrowing",
                        "round_broad": i,
                        "round_narrow": j,
                        "broad_evidence": pi.detected[RelationType.HYPERNYM].evidence[:1],
                        "narrow_evidence": pj.detected[RelationType.HYPONYM].evidence[:1],
                    })

        merged = self._merge_conversation_text(rounds)

        return ConversationSemanticTrajectory(
            per_round=per_round,
            merged=merged,
            growth={
                "new_types_per_round": new_types_per_round,
                "diversity_trend": diversity_trend,
            },
            cross_round_contradictions=contradictions,
        )

    def compare(
        self,
        profile: RelationProfile,
        expected_branches: set[RelationBranch] | None = None,
        expected_types: set[RelationType] | None = None,
    ) -> dict[str, set]:
        """Find gaps between detected relations and expected ones.

        Returns:
            {"missing_branches": set, "missing_types": set, "extra_types": set}
        """
        if expected_branches is None:
            expected_branches = {
                RelationBranch.TEMPORAL,
                RelationBranch.HIERARCHY,
                RelationBranch.OPPOSITION,
                RelationBranch.FUNCTIONAL,
            }
        if expected_types is None:
            expected_types = {
                RelationType.CAUSE_EFFECT,
                RelationType.HYPONYM,
                RelationType.COMPLEMENTARY_ANTONYM,
                RelationType.PURPOSE_MEANS,
                RelationType.CONDITION_CONSEQUENCE,
                RelationType.PROCESS_STAGE,
            }

        return {
            "missing_branches": expected_branches - profile.detected_branches,
            "missing_types": expected_types - profile.detected_types,
            "extra_types": profile.detected_types - expected_types,
        }
