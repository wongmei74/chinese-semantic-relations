# Chinese Semantic Relations

**Zero-LLM bilingual (Chinese/English) semantic relation analyzer.**

Extract 38 semantic relation types across 8 branches from text, generate targeted challenge strategies for gaps, and score semantic quality — all with regex-based analysis, no LLM API calls needed.

## Features

- **38 Relation Types** across 8 branches (Equivalence, Hierarchy, Approximation, Semantic Distance, Opposition, Temporal/Causal, Functional, Evaluative)
- **Bilingual** — works on Chinese and English text simultaneously
- **Zero LLM** — pure regex + rules, no API keys or costs
- **Zero Dependencies** — Python 3.10+ standard library only
- **Challenge Strategy Generation** — maps semantic gaps to difficulty-graded probe questions (L1 引导 → L2 正常 → L3 高压)
- **Quality Scoring** — evaluates semantic richness, composition depth, and anti-patterns on a 0-25 scale
- **Multi-round Conversation Analysis** — tracks semantic growth and detects cross-round contradictions
- **Pluggable Persona System** — inject your own role/persona resolver for AI training platforms

## Quick Start

```bash
pip install chinese-semantic-relations
```

```python
from chinese_semantic_relations import SemanticRelationEngine

engine = SemanticRelationEngine()

# Analyze text → find gaps → generate challenge questions
plan = engine.plan_challenge(
    "我们的AI平台帮助所有学生提高成绩",
    difficulty_level=2,
)
print(plan.instruction)
# → "你说'所有用户都需要这个'——真的是所有用户吗？
#    哪类用户最迫切？优先级怎么排？用什么标准判定？"

print(plan.probes[0].gap_description_zh)
# → "定义过于宽泛——未将概念具体化到可操作级别"

# Score semantic quality of a response
score = engine.evaluate_response(
    "因为K12赛道在线渗透率从15%提升到45%，"
    "具体来说聚焦三四线城市初中生数学。"
    "与现有产品不同的是我们用知识图谱实现个性化出题。"
)
print(f"Score: {score.total}/25")  # e.g. Score: 14.2/25
print(score.recommendations)       # actionable improvement suggestions
```

## Relation Taxonomy

| Branch | Types | Key Relations |
|--------|-------|--------------|
| Equivalence | 4 | Synonym, Logical Equivalence, Coreference |
| Hierarchy | 5 | Hypernym (overgeneralization), Hyponym (specificity), Part-Of |
| Approximation | 4 | Analogy, Metaphor, Near-Synonym |
| Semantic Distance | 4 | Same-Field, Frame-based, Collocation |
| Opposition | 5 | Antonym, Contradiction, Converse |
| Temporal/Causal | 6 | Cause-Effect, Condition, Purpose-Means, Process Stage |
| Functional | 5 | Tool-Purpose, Agent-Action, Action-Result |
| Evaluative | 4 | Vague Quantifier, Positive/Negative Connotation |
| Evidence | 1 | Evidence Citation (data, statistics, references) |

Each relation type includes:
- Formal properties (symmetric, transitive)
- Product relevance weight (0-1)
- Bloom's taxonomy minimum level
- Dimension affinity tags
- WordNet/ConceptNet alignment (where applicable)

## Advanced Usage

### Multi-round Conversation Analysis

```python
trajectory = engine.analyze_conversation([
    {"user_response": "因为用户需求大所以我们决定做这个产品"},
    {"user_response": "具体来说我们聚焦在教育行业的K12赛道"},
    {"user_response": "如果Q1获得种子用户就能在Q2推广"},
])

print(trajectory.growth["diversity_trend"])    # [2, 3, 4] — growing!
print(trajectory.cross_round_contradictions)   # detect inconsistencies
```

### Sector-Specific Analysis

```python
plan = engine.plan_challenge(
    "我们做一个SaaS产品",
    difficulty_level=2,
    sector="saas",  # also: education, fintech, healthcare
)
# → sector-specific relation expectations are merged in
```

### Custom Persona Resolver

```python
def my_resolver(dims: tuple[str, ...]) -> list[str]:
    mapping = {
        "metrics": ["data_analyst", "cfo"],
        "user_focus": ["ux_researcher"],
    }
    return [p for d in dims for p in mapping.get(d, [])]

engine = SemanticRelationEngine(persona_resolver=my_resolver)
plan = engine.plan_challenge("我们的产品很好", difficulty_level=2)
print(plan.probes[0].recommended_personas)  # ["data_analyst", "cfo"]
```

### Using Individual Components

```python
from chinese_semantic_relations import (
    SemanticRelationAnalyzer,
    SemanticChallengeStrategyMapper,
    SemanticResponseEvaluator,
)

# Just analysis
analyzer = SemanticRelationAnalyzer()
profile = analyzer.analyze("因为X导致Y，如果A则B")
print(profile.detected_types)  # {CAUSE_EFFECT, CONDITION_CONSEQUENCE}
print(profile.branch_diversity)  # 1

# Just strategy
mapper = SemanticChallengeStrategyMapper()
plan = mapper.plan_from_profile(profile, difficulty_level=3)

# Just scoring
evaluator = SemanticResponseEvaluator()
score = evaluator.evaluate(profile)
```

## Development

```bash
git clone https://github.com/wongmei74/chinese-semantic-relations.git
cd chinese-semantic-relations
pip install -e ".[dev]"
pytest
```

## Theoretical Foundation

The ontology is grounded in:
- **WordNet** — hypernym/hyponym/meronym hierarchy
- **FrameNet** — frame-based semantic roles
- **ConceptNet** — commonsense relation types
- **Bloom's Taxonomy** — cognitive complexity levels

Calibrated for structured reasoning assessment in product management, business strategy, and academic contexts.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Origin

Extracted from [ForgeMind](https://github.com/ForgeMind) — an AI pressure-forging platform for product manager training. The semantic relation engine is the core analysis module that powers zero-LLM question generation and response evaluation.
