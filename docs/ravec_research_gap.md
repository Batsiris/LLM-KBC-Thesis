# RAVEC: research gap validation and experimental roadmap

Review date: **14 November 2026**. Repository inspected at
`04e0a39fcbfdf8379795bdfac3ccd2b2b18da739`; working tree initially clean.
RAVEC is a provisional research name, not an implemented or validated method.
This report creates no inference, retrieval, training, or evaluation implementation.

Evidence labels used throughout:

- **[V] Verified:** supported by the linked primary paper, official repository,
  benchmark website, or directly inspected local source. This verifies what a
  source reports, not independent reproduction of its empirical claims.
- **[P] Proposed/interpretation:** our design, recommendation, or inference from
  the evidence; not an established result or a confirmed novelty claim.
- **[U] Unresolved:** unavailable, inconsistent, not inspected sufficiently, or
  requiring experimental/author confirmation. **[U-user]** denotes researcher-
  supplied results whose underlying artifacts were not available for this audit.

Sources were checked through accessible publisher PDFs, author resources, and
official benchmark pages. This is a scoped primary-source review, not a systematic
review proving the absence of all prior work. Score discrepancies and access
limitations are retained rather than resolved by assumption. NR means not reported
in the inspected material, not zero or evidence that a feature is absent.

## 1. Executive research summary

**[P] Recommendation:** investigate a small, budgeted **verify-versus-discover
decision policy for multi-answer completion**, with separately calibrated estimates
of candidate correctness and the usefulness of another evidence search. Require
an explicit precision constraint and evaluate recovered answers, damaged answers,
and computation together. Treat this as a falsifiable extension of existing
systems, not a novel architecture merely because it combines RAG and verification.

**[V] The largest novelty obstacle is COVER-KBC.** Its published framework already
has typed relation programs, provenance-bearing model observations, verification,
residual search-need estimates, and query-state-dependent action selection under
a budget. Adaptive control is therefore not missing from the closest KBC work.
[COVER-KBC, §§3–4](https://aclanthology.org/2026.akbc-1.37.pdf)

**[V] The official LM-KBC 2026 competition is closed-book and forbids additional
training.** Retrieval-assisted RAVEC and the existing QLoRA condition can use its
data/scorer for research but are outside that competition's information/training
setting. Do not present them as eligible official-track submissions or compare
their scores as evidence of architectural superiority over closed-book entries.
[Official rules](https://www.akbc.ws/2026/shared-task.html#rules),
[organizers' paper](https://aclanthology.org/2026.akbc-1.28.pdf)

### Repository and experimental starting point

**[V-local]** Inspected README, `pyproject.toml`, `docs/evaluation.md`, prediction
validation, zero/few-shot and QLoRA inference entry points, and the vendored scorer.
Existing modules already separate generation, flat-string prediction validation,
and benchmark evaluation. `zero_shot._run_prepared()` preserves raw generations,
parse-failure abstentions, coverage checks, and completion metadata. Preserve those
contracts. The local `results/` contains only `.gitkeep`; no experiment reports
were independently re-scored. See [evaluation documentation](evaluation.md),
[prediction I/O](../src/llm_kbc/prediction_io.py), and
[shared inference runner](../src/llm_kbc/zero_shot.py).

**[U-user]** The following 475-example validation results are supplied by the
researcher, not results generated or independently verified in this review:

| Condition | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|
| Zero-shot v2 | 0.5767 | 0.2906 | 0.2217 |
| 1-shot | 0.5676 | 0.3723 | 0.3354 |
| 3-shot v1 | 0.5992 | 0.3841 | 0.3408 |
| 5-shot | 0.6458 | 0.3605 | 0.3425 |
| QLoRA | 0.3403 | 0.4185 | 0.2898 |
| 5-shot union QLoRA | 0.3241 | 0.4661 | 0.3000 |

The union reportedly adds 381 raw candidates and yields a net 77 additional TPs.
**[P]** This motivates studying candidate selection and recovery, but does not
establish reliable knowledge, useful calibrated uncertainty, or a new algorithm.
Raw candidate counts are not official normalized candidate counts; net TP gains
do not identify how many added candidates are individually correct. Alias
matching and numeric tolerance can change that attribution. Do not infer a
77/381 marginal precision without an explicit matched-candidate audit.

## 2. Verified related-work comparison matrix

### 2.1 The five requested systems

**[V unless marked U]** All five use LM-KBC 2026 public train/validation data and
report official hidden-test results; model observations described as evidence
are not retrieved documents. The matrix summarizes each paper's contribution and
setting; it does not imply that the implementations have been run here.

| Paper and primary source | Models, inference, and external information | Generation, selection, verification, routing, and constraints | Limitations and RAVEC overlap |
|---|---|---|---|
| [COVER-KBC](https://aclanthology.org/2026.akbc-1.37/), PDF §§3–4 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`; frozen, greedy, NF4/BF16; one A100. No external factual retrieval or training. | Multiple acquisition views; independent-support groups; blind VALID/INVALID/UNKNOWN verification with content-free calibration; contradiction/disagreement penalties. Query-state controller chooses views, facets, verification, reverse checks, or stopping. Typed cardinality/nullable/numeric finalization, not a general OWL reasoner. | Strongest architectural overlap. Completeness control and targeted discovery already exist. Parametric evidence is correlated; open-set recall and venue capacity remain difficult. [PDF](https://aclanthology.org/2026.akbc-1.37.pdf) |
| [REAP](https://aclanthology.org/2026.akbc-1.31/) | Main: Mistral-Small-24B-Instruct-2501; comparisons: Llama-3.1-8B-Instruct and Gemma-2-9B-it. Closed-book; no tuning; vLLM TPU/BF16 experiments. | Relation-specific reasoning, geographic border scan, chronological award passes and an empty-set gate; deterministic parsing with LLM extraction/retry fallback. Primarily fixed relation programs and format-conditioned fallback, not an external-evidence controller. No general ontology engine described. | Already decomposes large answer sets and separates elicitation from serialization. Its parsing/repair policy differs from our strict baseline. [PDF §§3–4](https://aclanthology.org/2026.akbc-1.31.pdf) |
| [Relation-Specific Elicitation](https://aclanthology.org/2026.akbc-1.45/) | Base `gemma-4-31B`; gemma-2 reference configuration. No inference-time retrieval/training; offline Wikidata-derived area demonstration bank and auxiliary development set are explicitly disclosed. | Many-shot completion, sampled pools, relation-specific vote thresholds and numeric clustering; venue disambiguation; city vote-share abstention. A separate yes/no city gate is inactive in the reported ablation. Not query-state adaptive retrieval. Border reciprocity was tried and rejected. | Static elicitation/aggregation and abstention are prior art. Do not describe its information access as exclusively official train labels. Source versions disagree on scores. [PDF §§1,4–5](https://aclanthology.org/2026.akbc-1.45.pdf) |
| [CoRe](https://aclanthology.org/2026.akbc-1.42/) | Gemma-4-31B-it comparison; final Qwen3.5-27B. Closed-book; cached rationalized training exemplars; greedy generation. | Routes relations to direct answers or CoT; five hash-selected exemplars; parsing; numeric parse failures use the training median. Routing is relation-level, not a learned per-query action policy. No external-document verification or formal ontology described. | Useful static-routing control. Its forced numeric fallback is a methodological difference from our abstentions. Batch composition and token budgets confound some comparisons. [PDF §§3–5](https://aclanthology.org/2026.akbc-1.42.pdf) |
| [Coverage Is Not Knowledge](https://aclanthology.org/2026.akbc-1.33/) | Base Gemma-4-31B, vLLM, fixed-seed sampled completion channels; no retrieval/training; paper reports one H100 94GB. | Fixed train-only demonstrations, recitation/other elicitation frames, per-relation vote-share thresholds; numeric tolerance-window consensus. Abstention emerges from selection. Permuted-gold nulls challenge interpretation of numeric candidate coverage. No external-evidence or general ontology subsystem described. | A strong result is not proof that larger candidate pools represent knowledge. RAVEC must beat chance-corrected and cost-matched controls, not only union recall. [PDF §§4–6](https://aclanthology.org/2026.akbc-1.33.pdf) |

### 2.2 Metrics, split identity, and source conflicts

**[V]** The official final leaderboard is the authority for *ranked competition
submissions*, whereas a paper may report a different configuration. These scores
are contextual published results, not comparisons against our validation runs.
[Official final leaderboard](https://www.akbc.ws/2026/AKBC%20Shared%20Task%202026%20Final%20Leaderboard.pdf)

| System | Verified validation result | Published/official hidden-test result | Reproducibility qualification |
|---|---|---|---|
| COVER-KBC | F1 0.560 versus same-backbone direct 5-shot 0.492 | Macro P/R/F1 **0.7268 / 0.6055 / 0.5878**; rank 10 | [Public code](https://github.com/vquclinh/cover-kbc), Apache-2.0 shown by repository. Released profile has additional final relation layers. |
| REAP | Mistral F1 about 0.65; repository gives 0.645 | Paper rounds F1 to 0.62; leaderboard **0.6179**, rank 9. P/R numerical values not tabulated in inspected paper tables. | [Public code](https://github.com/yammdd/AKBC-Shared-Task-2026). License for code reuse remains **[U]**; parsing retries are part of the method. |
| Relation-Specific Elicitation | PDF Table 1: P/R/F1 **0.773 / 0.702 / 0.6843** | PDF: F1 **0.6961** | HTML abstract instead says validation 0.6975; live repository claims test 0.7121/rank 1. Its team is absent from the final ranked table. Do not silently combine these versions or treat its README rank as the final official rank. |
| CoRe | PDF Table 3: 0.542 routing, 0.548 with rewritten company prompt; PDF abstract says 0.546; HTML says 0.548 | Table 2 and leaderboard: **0.5408**, rank 12 | [Code link in official leaderboard](https://github.com/MightBeFonias/CoRe) could not be fetched. Exact validation variant and licensing remain **[U]**. |
| Coverage Is Not Knowledge | No single validation aggregate transcribed; do not infer one from the test score | F1 **0.7060**, rank 1; P/R not supplied by the final leaderboard | [Public code](https://github.com/thylinao1/lm-kbc-2026); README declares Apache-2.0. Verify LICENSE/NOTICE and pin a commit before copying. |

**[V] Additional ranked comparator:** [MAVE](https://aclanthology.org/2026.akbc-1.38/)
uses Mistral-Small-3.2-24B, multiple question views with one vote per view, and
relation-specific emission rules. Its test F1 is **0.6908**, rank 2. This reinforces
that multi-view aggregation and relation thresholds are established. The
[paper PDF](https://aclanthology.org/2026.akbc-1.38.pdf) is accessible;
its [advertised repository](https://github.com/WSE-research/MAVE-LM-KBC-2026) was
not accessible through the research browser. License and runnable release: **[U]**.

### 2.3 Broader methodological precedents

**[V]** The following are published precedents with different tasks/protocols.
Their numbers are not LM-KBC scores. A missing set-P/R/F1 entry means that the
paper uses different metrics or a full result was not transcribed here.

| Work | Verified models, data, and setting | Mechanism, metrics, resources, and overlap |
|---|---|---|
| [Self-RAG, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf) | Trained Llama2-based 7B/13B models; PopQA, TriviaQA, PubHealth, ARC-Challenge, ASQA and biography generation; external retrieval permitted. | Reflection tokens control retrieval, relevance, support and utility; segment-level selection. Reports accuracy, factuality and citation P/R, not complete KB-set F1. The [paper text](https://arxiv.org/html/2310.11511) reports PopQA accuracy 55.8 for 13B. [Code/models/data](https://selfrag.github.io/). Adaptive evidence-supported selection is prior art; prompting Qwen with critique instructions is not a reproduction of trained Self-RAG. |
| [FLARE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.495/) | `text-davinci-003`; 2WikiMultihopQA, StrategyQA, ASQA, WikiAsp; external text retrieval. | Low-confidence upcoming text triggers query formulation and regeneration. On 2Wiki, FLARE-direct P/R/F1/EM = 59.1/62.6/59.7/51.0 (percent). These are QA answer metrics, not entity-set metrics. [Paper](https://aclanthology.org/2023.emnlp-main.495.pdf), [code](https://github.com/jzbjyb/FLARE). Overlap: when/what to retrieve. Does not establish a complete-set stopping guarantee. |
| [Adaptive-RAG, NAACL 2024](https://aclanthology.org/2024.naacl-long.389/) | FLAN-T5-XL/XXL and GPT-3.5-Turbo-Instruct; single-hop SQuAD/NQ/TriviaQA and multi-hop MuSiQue/HotpotQA/2Wiki evaluations. | Outcome-labelled small classifier chooses no, single, or iterative retrieval. Reports EM/F1/accuracy, steps and time; e.g. FLAN-T5-XL HotpotQA F1 53.82 in Table 2. [PDF](https://aclanthology.org/2024.naacl-long.389.pdf), [code](https://github.com/starsuzi/Adaptive-RAG). A learned query-level router or outcome-derived action labels are not novel by themselves. |
| [RARR, ACL 2023](https://aclanthology.org/2023.acl-long.910/) | Main research/editor model PaLM 540B; GPT-3 implementation also studied. Generated passages from NQ, StrategyQA and QReCC; 150 development/150 test passages per model/task combination; web search. | Research questions, evidence agreement and bounded revision preserve input while improving attribution. Reports AIS, preservation and their F1-AP, not set P/R/F1; Table 4 gives NQ automatic attribution 45.6→54.9 and preservation 89.6. [PDF](https://aclanthology.org/2023.acl-long.910.pdf), [code](https://github.com/anthonychen/RARR). Evidence verification/correction is established; its preservation objective is not missing-object recovery. |
| [Chain-of-Verification, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.212/) | Llama 65B few-shot; Llama2-70B-Chat comparisons; Wikidata/Wikilist list questions, closed-book MultiSpanQA and long-form generation. | Independently answered verification questions reduce anchoring. Table 1 Wikidata precision increases 0.17→0.36 for the two-step variant while average correct items fall 0.59→0.38: precision gains need not preserve completeness. [PDF](https://aclanthology.org/2024.findings-acl.212.pdf). Official executable reproduction resources were not established **[U]**; prompts are described. No retrieved documents are required in its closed-book experiments. |
| [KICGPT, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.580/) | RotatE retriever plus `gpt-3.5-turbo`; FB15k-237 and WN18RR official link-prediction splits, with validation tuning. Uses graph training triples; no LLM fine-tuning. | Reranks retrieved entity candidates using structural demonstrations. Table 1 gives filtered MRR **0.412** on FB15k-237 and **0.549** on WN18RR, with Hits@1,3,10; these are not set P/R/F1. [PDF §§3–4](https://aclanthology.org/2023.findings-emnlp.580.pdf); runnable code/license not verified **[U]**. Already combines retrieval and LLM selection, but ranks a known entity universe rather than solving abstention/open-set completion. |
| [KGC-RAG, LM-KBC 2024](https://ceur-ws.org/Vol-3853/paper13.pdf) | Meta-Llama-3-8B-Instruct plus all-MiniLM-L6-v2; 2024 train/val/test, external Wikipedia pages and linked pages, Wikidata entity mapping. | URL similarity and LLM relevance filtering precede context-ranked extraction; linked entities are checked against relation ranges. Macro P/R/F1: validation **0.799/0.801/0.695**, test **0.792/0.810/0.698**. [Public code](https://github.com/jaejeajay/LM-KBC2024), runnable release/license not audited **[U]**. Already combines broader evidence acquisition and type checks. No query-state verify-versus-discover controller demonstrated; source/gold disagreement limits scores. |
| [Extracting Multi-valued Relations, RepL4NLP 2023](https://aclanthology.org/2023.repl4nlp-1.12/) | BERT-scale masked-LM probing on seven multi-valued Wikidata relations; train/development/test study, not LM-KBC 2026; no web evidence. | Rank-then-select with learned relation thresholds. Best reported P/R/F1 = **54.1/50.8/49.5%**. [PDF](https://aclanthology.org/2023.repl4nlp-1.12.pdf); author resources are linked from [Singhania's page](https://people.mpi-inf.mpg.de/~ssinghan/). Domain-aware prompting, cardinality estimation and verification were investigated before RAVEC; single-token scope limits transfer. |
| [ReWiSe, LM-KBC 2025](https://ceur-ws.org/Vol-4041/paper8.pdf) | Qwen3-8B inference; Llama3.3-70B constructs synthetic CoT demonstrations offline. Official 2025 test, no inference-time retrieval or model tuning. | Relation-wise entity self-consistency thresholds; 20 samples in strongest configuration. Macro P/R/F1 = **0.546/0.482/0.444**. [Code](https://github.com/Lama-West/ReWiSe), linked by [organizers](https://lm-kbc.github.io/challenge2025/). Already explicitly controls the P/R trade-off; extra sampling is costly. |
| [Self-Retrieval + Divide-and-Conquer, LM-KBC 2025](https://ceur-ws.org/Vol-4041/paper11.pdf) | Qwen3-8B; official 2025 benchmark; self-generated descriptions rather than external RAG. | Describe-then-extract and decomposition/union for high-cardinality answers. Organizer test F1 **0.405**. [Code](https://github.com/JingboHH/LM-KBC-ISWC-2025/), [ranking](https://lm-kbc.github.io/challenge2025/). Demonstration/candidate decomposition and completion overlap; do not mislabel self-retrieval as document retrieval. Further P/R and exact reproduction settings **[U]**. |
| [KARMA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/517f9b9c227b9dd51dba4560f37165ed-Paper-Conference.pdf) | GPT-4o, DeepSeek-v3 and GLM-4; 1,200 PubMed articles in three domains, rather than a standard held-out LM-KBC split. External articles and existing KGs are input. | Nine-agent extraction, schema alignment, conflict resolution and verification. Reports up to 38,230 added entities, 83.1% **LLM-verified** correctness and 18.6% conflict-edge reduction; not gold set P/R/F1. Paper promises supplementary/open code; runnable licensed release not established **[U]**. Broad evidence-plus-schema enrichment is already published; nine agents are not a reason to add them here. |
| [OntoSCPrompt, IEEE ICSC 2025](https://www.inf.uni-hamburg.de/en/inst/ab/sems/resources/jiang-et-al-icsc-2025-ontology.pdf) | LongT5 with learned hybrid prompts; CWQ, WebQSP, LC-QuAD 1.0, plus cross-KG DBLP/CoyPu evaluations; graph access required. | Separates SPARQL structure/content, ontology conditioning and constrained decoding. Table IV: LC-QuAD P/R/F1 **92.9/68.8/79.1%**, WebQSP Hits@1 **73.8%**; also reports query EM. [Code](https://github.com/LongquanJiang/OntoSCPrompt). Ontology guidance is established, but executable graph queries differ from validating newly proposed facts. |
| [Text2KGBench, ISWC 2023](https://arxiv.org/html/2308.02357) | Vicuna-13B and Alpaca-LoRA-13B; ontology-conditioned extraction from supplied TekGen/WebNLG sentences; labelled test and unseen-sentence subsets. | Measures extraction P/R/F1, ontology conformance and hallucinations. Vicuna on all Wikidata-TekGen: **0.38/0.34/0.35**, conformance 0.83. [Code/evaluator](https://github.com/cenguix/Text2KGBench), [dataset](https://zenodo.org/records/7916716). Formal conformance does not imply factual correctness or complete world knowledge. |
| [ALCE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/) | Retrieval-grounded long-form answers on ASQA, QAMPARI and ELI5; multiple open/API models. Exact model-specific scores not transcribed **[U]**. | Benchmarks correctness and citation precision/recall, including multi-answer coverage. [Paper](https://aclanthology.org/2023.emnlp-main.398.pdf), [code/data](https://github.com/princeton-nlp/ALCE). Supports separate evidence-quality and answer-completeness evaluation. Citation recall is coverage of generated claims, not necessarily recall of every correct entity. |

## 3. Critical overlap analysis: COVER-KBC versus RAVEC

**[V]** COVER-KBC's §3.4 is already query-state adaptive, not merely a static
relation router: legal actions are scored using expected yield, gap closure,
uncertainty, cost and redundancy. Its relation contracts determine allowed actions
and stopping. Its core caps are 12 calls/6,000 tokens, with separately capped
relation escalations. Its verifier omits generator rationale/support counts and
distinguishes unknown from explicit contradiction.
[COVER-KBC paper, §§3.3–3.5 and Appendix A](https://aclanthology.org/2026.akbc-1.37.pdf)

| Proposed RAVEC ingredient | Prior art already identified | Defensible interpretation |
|---|---|---|
| Adaptive retrieval/inference | COVER-KBC; FLARE; Self-RAG; Adaptive-RAG | **[P]** Not novel generically; external-document action utility must outperform matched routing controls. |
| Evidence accumulation and verification | COVER-KBC; RARR; CoVe; KARMA | **[P]** Replacing model observations with retrieved passages changes information access, not automatically the algorithmic contribution. |
| Recover missing objects | COVER-KBC facet/missingness search; REAP chronological passes; 2025 divide-and-conquer | **[P]** Recovery itself is established; isolate *which unresolved gap deserves a search* and how unsupported additions are prevented. |
| Relation/type/cardinality rules | COVER-KBC contracts; OntoSCPrompt; Text2KGBench; KARMA | **[P]** Not new as a bundle. A typed, evidence-qualified action mechanism would need an independent ablation. |
| Higher P and R | Relation thresholds and self-consistency throughout this literature | **[P]** A Pareto improvement under matched information/cost is an empirical result, not a novelty definition. |

**[P] Novelty verdict:** the originally stated architecture is insufficiently
distinct. Do not claim the first adaptive, evidence-centric, verification-and-
completion KBC system. A controller learned from outcomes is also insufficient
alone because Adaptive-RAG already uses outcome-derived routing labels. Novelty
must reside in a precisely defined set-completion decision problem, estimator,
or evaluation result that survives the controls below.

## 4. Candidate research gaps and novelty assessment

All three directions are **[P] candidate contributions**, not verified gaps.

| Direction | Existing overlap | Narrow contribution worth testing | Feasibility | Falsifier and isolating ablations |
|---|---|---|---|---|
| A. Precision-constrained verify-versus-discover policy | COVER-KBC already chooses verification/discovery/stopping; Adaptive-RAG learns routing from outcomes. | Estimate **separate action-conditioned changes in accepted TP and FP**, including new-object yield, then allocate a fixed budget at each subject–relation state under a calibrated precision target. Contrast this with a generic uncertainty or question-complexity router. | Medium: a small deterministic linear/logistic policy or threshold table; requires train-only action traces and enough calibration data. No policy-training framework initially. | Fails if a fixed schedule, relation router, or COVER-style heuristic matches its cost/P/R frontier. Remove cost, correctness features, recovery-yield features, and relation identity separately; compare identical legal actions. |
| B. Evidence-gap-directed recovery with protected verified answers | REAP/decomposition, COVER-KBC facet discovery, RARR revision and KARMA enrichment. | Compare targeted search for **uncovered evidence facets** with another broad search. Preserve accepted answers unless new explicit contradiction justifies removal; record each added/deleted object and supporting span. Test provenance-diverse versus duplicated evidence. | Medium: deterministic facet templates for awards and borders, plus a generic corpus search. True world completeness is not observable. | Fails if broad extra retrieval at the same budget recovers as much, or precision protection loses correct answers. Ablate targeted queries, contradiction-only removal and source grouping; compare filter-only and recovery-only. |
| C. Evidence-qualified constraints as action signals | Typed COVER contracts; schema alignment/validation literature. | Distinguish a **hard inconsistency**, an unresolved type/time/unit conflict and mere missing evidence. Route conflicts to specific retrieval rather than silently deleting everything that fails a soft check. | High for six explicit contracts; low/medium for transferable entity typing without a pre-existing KG. | Fails if simple post-filtering performs as well or if constraints mostly suppress valid answers. Compare hard rejection, soft features, routing features, type-only, numeric-only and no constraints. |

**[P] Alternatives if A–C do not survive prior-art or experimental checks:**

1. Make the thesis a controlled **precision/completeness/cost study** across
   closed-book, retrieval and adapter conditions, with chance-corrected numeric
   pool coverage and evidence audits as the contribution.
2. Investigate **calibration transfer under relation and empty-answer-rate shift**:
   can a threshold/policy frozen on one benchmark maintain its operating point on
   new relations? A clear negative result is useful; do not relabel it a new SOTA
   architecture.
3. Study evidence availability versus selection failure on a fixed candidate pool,
   separating model ignorance, retrieval failure, parsing loss and verifier loss.

**[V/P]** KGC-RAG adds a particularly direct precedent for C: retrieved evidence
plus relation-range filtering is already evaluated on LM-KBC. Consequently,
“RAG plus ontology/type validation” is not a sufficient contribution either.
[KGC-RAG, §3.2](https://ceur-ws.org/Vol-3853/paper13.pdf)

## 5. Recommended primary research contribution

**[P] Primary question:** *At a fixed evidence and inference budget, can a
subject–relation policy that separately estimates the benefit of verifying current
candidates and discovering new objects improve recall at a predeclared precision
operating point compared with static and uncertainty-only policies?*

The candidate mechanism is a small **action-value estimator for object sets**,
not a new generic RAG stack. Candidate correctness and prospective recovery yield
must be different signals: high confidence in two names does not establish that
an award has only two recipients. Offline training traces can execute the same
bounded actions for each eligible training query and measure their consequences
using *training gold only*. Exclude demonstration identities and cross-fit the
policy/calibration so an answer is never provided as its own demonstration.

Define the operating target and noninferiority margin before the final study.
For example, maximize official macro recall with macro precision no worse than a
frozen matched RAG baseline minus a prespecified margin, subject to a call/token
budget; use macro F1 as a co-primary summary and show the full trade-off curve.
Also constrain/report nonempty-output precision and empty-case false positives,
because official empty-prediction precision equals one. Do not claim a formal
precision guarantee from uncalibrated verifier confidence or a small development
sample. Dataset shift invalidates naive calibration-transfer assumptions.

**[U]** Whether this estimator is sufficiently distinct from all existing
completeness-aware controllers remains to be established. Approval should authorize
a controlled hypothesis test, not a claim of novelty in advance.

## 6. Falsifiable research hypotheses

**[P]** Freeze datasets, budgets, corpus, prompts, controller fits, thresholds,
and the primary contrast before final labels are inspected.

| Hypothesis | Measurement and failure condition |
|---|---|
| H1: Separate verify/discover values improve the operating frontier. | Adaptive versus static relation routing and COVER-style state heuristics at equal budgets. Require a positive paired recall difference while meeting the prespecified precision margin; otherwise H1 is unsupported. Report F1 rather than reconstructing it from macro P/R. |
| H2: Targeted recovery adds supported answers beyond filtering. | Track newly matched TPs, added FPs, removed TPs, evidence support and net row F1. Compare against broad additional retrieval with identical calls/passages/tokens. If extra budget explains the gain, reject the targeted-recovery explanation. |
| H3: Constraint-triggered retrieval helps more than constraint filtering. | Equal evidence/models, with routing-versus-filtering/no-constraint variants. Require gains beyond syntactic validity or empty-output changes; otherwise do not attribute gains to ontology reasoning. |
| H4: The policy transfers. | Freeze it on LM-KBC 2026 training/development and test on a separately reserved benchmark/model. No target-test threshold tuning. If only retuning works, claim adaptation, not zero-shot generalization. |
| H5: Complementary candidate pools contain usable headroom. | Analyze the existing union offline, with the official matcher, source-aware additions and relation-stratified numeric permutation controls. If gains disappear after chance/magnitude controls or supported recovery is negligible, prioritize knowledge acquisition rather than selection sophistication. |

## 7. Dataset and cross-benchmark comparison matrix

**[V]** Task compatibility is not protocol equivalence. Public test *inputs* do
not imply public gold. **[U]** Historical evaluation-service availability and exact
current test-label contents were not established for every edition; never score
placeholder labels. The additional benchmark should use a frozen, separately
audited adapter rather than the 2026 matcher by default.

| Benchmark / official sources | Schema, splits, labels and license | Evaluation, retrieval setting, published results | Suitability and caveats |
|---|---|---|---|
| [LM-KBC 2026](https://github.com/lm-kbc/dataset2026), [rules](https://www.akbc.ws/2026/shared-task.html) | Subject/relation JSONL; gold nested object-alias groups; prediction flat strings. Local reviewed condition: 477 train / 475 val / 475 unlabeled test. Repository Apache-2.0; pinned evaluator provenance is retained locally. | Alias-normalized one-to-one string matching; numeric 5% tolerance; row-macro and count-micro P/R/F1. Official track prohibits retrieval/training; winning final F1 0.7060. | Best in-project benchmark for empty/many/numeric cases. RAG/QLoRA are explicitly research extensions. Public validation is already development data. |
| [LM-KBC 2023](https://lm-kbc.github.io/challenge2023/), [repo](https://github.com/lm-kbc/dataset2023), [overview](https://ceur-ws.org/Vol-3577/paper0.pdf) | 21 relations in paper/repository, despite a website sentence saying 20. Train/val/test; mostly 100 subjects per relation with smaller geographic/chemical subsets. Subject labels/IDs, object labels/IDs; official predictions require `ObjectEntitiesID`. Train/val gold available; test subjects and challenge scoring released, current test-label accessibility **[U]**. Data CC BY 4.0 per organizers. | ID-based set P/R/F1 with explicit empty cases; exact aggregation/scorer revision must be pinned before reuse. Open track explicitly permits retrieval; small-model track forbids contextual augmentation. Open-track winner LLMKE: P/R/F1 **0.715/0.726/0.701**. | **Recommended first additional benchmark:** new relations, different cardinalities, numeric counts, entity disambiguation and a retrieval-permitted track. Audit overlap and historical fact dates; entity linking becomes a separately measured component. |
| [LM-KBC 2024](https://github.com/lm-kbc/dataset2024), [site](https://lm-kbc.github.io/challenge2024/), [overview](https://ceur-ws.org/Vol-3853/paper0.pdf) | Five relations; repo table gives 63 borders + 100 death + 100 episode-count + 10 awards + 100 exchanges = 373 rows per split. ID predictions; train/val/test files; withheld challenge scoring. Data repository CC BY 4.0. | Macro P/R/F1, ID outputs; inspect exact numeric/scorer handling before reuse **[U]**. Overview requires self-contained inference/no web-search calls and <10B models. Final winner score **0.9224**. | Episode counts add numeric generalization, but four relations overlap our task. Do not assume unrestricted RAG: the winning lookup-based setting and inference rules need reconciliation before comparison. |
| [LM-KBC 2025](https://github.com/lm-kbc/dataset2025), [site](https://lm-kbc.github.io/challenge2025/) | Six relations. Repo declares 478 train / 478 val / 477 test from its table; verify actual bytes before use. Gold `ObjectEntities` flat strings; optional IDs persist in examples. CC BY 4.0 repository. Labeled train/val; public test input release, current gold accessibility **[U]**. | Fixed Qwen3-8B; no fine-tuning or external RAG. [Scorer](https://raw.githubusercontent.com/lm-kbc/dataset2025/master/evaluate.py) uses string equality after prediction stripping and numeric tolerance, not 2026 alias matching. ReWiSe test F1 **0.444**. | Easy task-shape transfer but weak independence: overlap with 2026 must be audited. RAG would be a research extension here too. Never compare editions' scores as the same task. |
| [KAMEL](https://github.com/JanKalo/KAMEL), [paper](https://akbc.ws/2022/assets/pdfs/15_kamel_knowledge_analysis_with_.pdf), [author HF card](https://huggingface.co/datasets/LeandraFichtel/KAMEL) | 234 relations; multi-token names, aliases and literals; at most 10 answers. Paper and split script specify **1,000/200/200 per relation**; HF card instead states **200/100/100**. GitHub grouped schema includes `index`, `sub_label`, `obj_uri`, `obj_label` alias dictionaries; HF card describes a different flat record schema. Card declares **CC BY-NC 4.0**, but links LAMA's license rather than a KAMEL-specific license file. | Paper: per-query alias-aware P/R/F1, averaged within relations then across relations; no 2026 numeric-tolerance equivalence established. Historical OPT-13B 10-shot P/R/F1 **17.76/17.48/17.62%**; not a verified current SOTA. Public split resources advertised; no modern closed-book competition rules. | Strong broader transfer candidate **after release/schema/license audit**. Do not run untrusted legacy dataset scripts. External retrieval is a new declared condition; source-Wikipedia availability and incomplete KB gold affect interpretation. |
| [KnowledgeNet](https://github.com/diffbot/knowledge-net), [paper](https://aclanthology.org/D19-1069/) | Document/passages with text offsets, property IDs and subject/object Wikidata URIs; five folds, train file and unannotated fifth-fold test. Repository MIT; source-text rights should be retained/audited separately. | Official span-exact, span-overlap and URI evaluation; PRF at fact/mention granularity. Supplied text is the evidence. Published baseline linked F1 **0.504**; repository later lists Diffbot Joint Model 0.726, without establishing a comparable peer-reviewed reproduction here. | Useful evidence-extraction stress test, **not a drop-in subject–relation world-completion benchmark**. A derived grouped-set task must be labelled as a new protocol. Numeric completeness is not established. |
| [Text2KGBench](https://zenodo.org/records/7916716), [repo](https://github.com/cenguix/Text2KGBench) | Sentence + ontology → triples (`sub`, `rel`, `obj`); train/test and published test gold, manually checked and unseen-sentence subsets. 13,474 TekGen and 4,860 WebNLG sentences. Paper says data CC BY 4.0; repository code Apache-2.0. Zenodo description inconsistently says “Attribution-ShareAlike” but abbreviates CC BY: verify artifact LICENSE before redistribution. | Official extraction P/R/F1 plus conformance/hallucination metrics. Supplied text, not closed-book probing. Historical Vicuna F1 0.35 on all TekGen; no current SOTA claim. | Good secondary test of evidence/ontology modules. Not a replacement for evaluating complete world-object sets; numbers/literals are not evaluated with the LM-KBC 2026 protocol. |

**[P] Recommended sequence:** use LM-KBC 2023 open-track data first, with untouched
subject/relation or relation-disjoint subsets reserved before implementation.
Retain its official ID scorer and report entity-linking errors separately. Audit
subject IDs, object-to-subject links and facts shared with all 2026 splits; spelling
normalization alone is insufficient. Use KAMEL as the broader second-stage
replication if the release discrepancies and license provenance can be resolved.
For an MSc-scale KAMEL subset, preregister selection by relation type/size only,
publish all included/excluded relation IDs and report it as a subset, not the full
benchmark score.

**[P] Independence warning:** a 2025/2026 edition change is not strong evidence of
generalization if queries/facts are reused. Public KAMEL/2023 labels also do not
guarantee exclusion from pretrained model data. Freeze a target evaluation split
before viewing its labels or adapting the controller. If no untouched label set
or organizer evaluation is available, call the outcome cross-validation or
development evaluation, not genuinely held-out final testing.

**[V/U] 2024 setting/version warning:** the winning
[Bara system paper](https://ceur-ws.org/Vol-3853/paper10.pdf) explicitly supplies
Wikidata SPARQL context and checks output identities against Wikidata; KGC-RAG
uses external Wikipedia retrieval and reports 68 border rows (378 total per
split), versus 63 (373 total) in the current official repository table. Both are
published systems, but these differences prevent inferring comparable scores from
the year alone. Resolve the apparent tension with the overview's no-external-call
wording and pin the actual data version before reproducing a 2024 result.

## 8. Published SOTA and reproducibility assessment

### A. Published-score comparison

**[P]** Maintain a contextual table containing dataset *hash/release*, split,
scorer revision, averaging, model, training, retrieval, demonstration provenance,
sampling budget and score source. Use the official final ranked leaderboard for
2026 ranks. Do not infer missing P/R from F1. Do not rank our Qwen validation
conditions alongside 24–31B hidden-test entries as if conditions were matched.
RAVEC retrieval results belong in a separate information-access stratum.

**[V]** Relevant official final test reference points are 0.7060 (Coverage),
0.6908 (MAVE), 0.6179 (REAP), 0.5878 (COVER-KBC) and 0.5408 (CoRe).
The [official leaderboard](https://www.akbc.ws/2026/AKBC%20Shared%20Task%202026%20Final%20Leaderboard.pdf)
does not corroborate the different live rank claim in Relation-Specific
Elicitation's [README](https://raw.githubusercontent.com/ruggsea/lmkbc2026-submission/main/README.md).
Use its PDF score as a paper-reported configuration, with the conflict visible.

### B. Reproduced-method comparison: three priorities

| Priority | Method and reason | Minimum faithful comparison and feasibility |
|---|---|---|
| 1 | **COVER-KBC:** closest adaptive architecture; accessible Apache-2.0 repository. | Freeze release/profile, relation contracts, controller, parsing and verification. First port the mechanism to Qwen with explicit deviations; call this a **method adaptation**, not reproduction of its published score. Then, if feasible, run its original Mistral/NF4 setup on matched data. Inspect final rescue layers and separate escalation costs. |
| 2 | **REAP:** simpler decomposition/gating baseline with public prompts/code. | Preserve chronological award passes, extraction/retry rules and empty gating. Qwen adaptation is feasible; original Mistral-2501 backend must be verified separately. Resolve code license before copying; alternatively independently implement the published procedure with attribution. Its permissive parser is part of its method, not our baseline's parser. |
| 3 | **KGC-RAG:** published retrieval-assisted LM-KBC system, with an 8B backbone and public implementation. | Freeze Wikipedia/link expansion, relevance filtering, entity mapping/range checks and data revision. Its smaller backbone makes a Qwen adaptation more practical than a 31B winner reproduction. License, corpus snapshots and protocol differences remain prerequisites. Reusing retrieval ideas on 2026 is a new research condition, not reproduction of its 2024 score. |

**[P]** Standard RAG and an Adaptive-RAG-style query router provide the additional
matched retrieval controls; they do not substitute for the closest KBC baseline.
For a higher-cost original-backbone SOTA reproduction, prioritize **Coverage Is
Not Knowledge**, freezing channel prompts, thresholds and numeric consensus after
license/NOTICE verification. If that is infeasible, retain the winning score as
context and use CoRe as a lightweight routing adaptation. MAVE is another strong
option if its release can be accessed and licensed.

**[V] Hardware constraint:** the official
[Mistral-Small-3.2 model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
states approximately **55GB GPU RAM** in BF16/FP16. It does not fit unquantized on
one A100 40GB. The Coverage paper reports approximately 62.5GB of BF16 weights.
Both [Mistral](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
and [Gemma-4-31B](https://huggingface.co/google/gemma-4-31B) model pages are reachable;
this does not establish compatibility with our pinned Transformers 4.57.1.
**[P]** Keep their reproduction environment separate; consider original published
quantization or larger/multiple GPUs without changing the thesis baseline.
Quantized ports must be labelled and compared at the same quantization.

### C. Controlled architectural comparison

**[P]** All causal architecture claims require identical backbone/revision,
candidate starting pool, adapter status, corpus snapshot, retriever, passage budget,
generation limits, decoding, context and evaluation identities. Compare separate
strata: closed-book/no training; retrieval/no training; adapter/no retrieval;
adapter+retrieval. Report both equal maximum budget and equal observed mean cost.
Count offline trace generation/policy fitting separately from inference. A policy
that uses more retrieval must not be credited for adaptivity without a matched
extra-retrieval control.

## 9. Proposed RAVEC architecture

Everything in this section is **[P] implementation-independent design**. The
blocks are established techniques; only the narrow action-value mechanism in §5
is a candidate contribution. Use one sequential controller and one generator,
not a multi-agent framework.

### 9.1 Candidate generation and identity

- Start with archived 5-shot candidates; optionally add the archived QLoRA union
  as a separately labelled condition. This isolates selection from new inference.
  Never load archived evaluation reports or gold into the inference process.
- Retain raw strings, generator/adapter IDs, prompt hashes, parse status and origin
  order. Preserve existing strict parsing and abstentions. Candidate identity and
  alias grouping must not depend on the gold alias sets.
- Entity linking, if needed for the external benchmark, is a separate component
  using a frozen entity index, with its own accuracy/cost accounting. Never obtain
  object candidates by querying benchmark gold or a held-out answer lookup table.

### 9.2 External evidence and provenance

- Begin with a frozen, licensed text corpus and deterministic lexical retrieval;
  add a dense retriever only if a matched experiment justifies it. Pin corpus,
  index, chunk boundaries, retriever/model versions, query templates and ranking
  tie rules. Use subject disambiguation and the relation definition in queries.
- For 2026, align factual claims to **1 July 2026**. A crawl after that date is
  not automatically valid evidence of that world state. Record publication,
  revision and retrieval dates; reject or flag temporally ambiguous assertions.
- Store document/revision ID, URL, content SHA256, text span offsets, query/rank,
  acquisition time, source type and duplicate/syndication group. A URL alone is
  insufficient provenance. Cache byte-identical documents and retrieved passages.
- Exclude benchmark repositories, submissions, generated predictions, leaderboard
  pages and gold-derived mirrors from the corpus. Do not equate retrieving a
  Wikidata answer field with independently verifying it. A direct-KB-lookup
  baseline, if studied, must be explicitly separated and permitted by that setup.
- Source reliability is a feature, not proof. Two copied pages are one provenance
  group. Corpus passages are untrusted data, not executable instructions.

### 9.3 Verification and targeted recovery

- Verify each `(subject, relation, candidate)` against selected spans using
  SUPPORTED / CONTRADICTED / INSUFFICIENT labels and cited offsets. Hide generator
  rationale and vote count from the verifier; include them only as separately
  ablated controller features. Calibrate on training traces, not validation labels.
- An unsupported candidate is not necessarily false. Preserve that distinction
  in audit logs even when it is withheld from the scoring prediction.
- For discovery, search uncovered facets: award-year/edition ranges or uncovered
  list sections; directional border checks; unresolved issuer status; competing
  venue identity/capacity interpretations. Facets are fixed from relation semantics
  and observed evidence, never missing gold items or known gold cardinality.
- Extract only newly supported objects, then pass them through the same verifier.
  Keep supported prior objects unless explicit evidence contradicts their identity,
  relation or temporal validity. Compare with unrestricted revision in an ablation.
- Evidence exhaustion or no new supported objects means **stop under budget**,
  not “the real-world set is proven complete.” A complete authoritative dated list
  can justify a stronger, separately recorded completeness signal.

### 9.4 Relation and ontology constraints

| Relation | Proposed constraint/evidence check; never a gold-derived rule |
|---|---|
| Borders | Country/territory identity consistent with benchmark definition; land adjacency, not maritime contact. Reciprocity is a soft cross-check and must not invent an unqueried fact. No globally fixed border count. |
| City of death | City-granularity location and evidence of death at the target date; no substitution of birthplace/residence. A living-status assertion can support a true empty answer; lack of search results cannot. |
| Capacity | Finite numeric answer in spectators, correct venue/location and dated configuration, targeting the benchmark's maximum published spectator capacity. Seated versus total capacity must be distinguished. |
| Area | Finite area in square kilometres, correct geographic referent and total-area scope. Record any explicit unit conversion and source units; do not infer decimal punctuation or transform unsupported values. |
| Award recipients | Correct award identity and recipient type, through the cutoff; works versus their recipients, predecessor/successor awards and rescinded awards require scoped evidence. No inferred maximum cardinality. |
| Exchanges | Valid exchange identity and issuer listing at cutoff, including the benchmark's depositary-receipt scope. Distinguish subsidiaries, delisting and private status. Multiple exchanges remain possible. |

The underlying 2026 definitions are **[V]** in the
[official dataset README](https://github.com/lm-kbc/dataset2026#relation-definitions).
The use of those definitions as decision features is **[P]**. Without explicit
type vocabulary, constraint representation and inference semantics, call this
*relation-aware semantic validation*, not a full ontology reasoner. Domain/range
conformance alone cannot prove a triple or an empty answer. Keep soft type
uncertainty separate from hard datatype violations; do not assume complete types.

### 9.5 Controller state, actions, objective, and stopping

State features: relation type; candidate count; generator disagreement; parse
status; calibrated support/contradiction/insufficient mass; independent-source
count; duplicate evidence fraction; entity ambiguity; temporal/unit conflicts;
facet coverage; recent verified discovery yield; remaining calls/tokens/time.
No gold count, validation TP/FP indicator, or evaluation report is a runtime feature.

Actions: accept/withhold an object; retrieve general evidence; retrieve evidence
for a disputed candidate; discover in an uncovered facet; verify new candidates;
resolve identity/unit/time conflicts; stop with a supported set or abstain.
Selection is deterministic given the recorded state and fitted parameters.

Fit separate training-only estimates `E[delta TP | state, action]` and
`E[delta FP | state, action]`, plus action cost and uncertainty. Their conversion
to an estimated set-level utility is itself a design choice to preregister and
ablate; additive candidate probabilities do not exactly optimize macro F1.
At deployment, choose the feasible positive-utility action under the precision
operating point and remaining budget. Stop on budget exhaustion, no legal useful
action, evidence saturation or a justified closed facet; log the reason.
“Correct” is an allowed offline *training target*, never a runtime oracle.

Initial experimental budget proposal: 1/2/4 retrieval rounds, at most five new
passages per round and a fixed total evidence-token allowance. Batch candidate
verification only within a declared context limit. Preserve 512 tokens for each
baseline-shaped object-array generation in the primary comparison. If recovery
requires chunked arrays or a larger output cap, make this a separate factor and
give controls the same opportunity. Record prefills as well as output tokens,
verifier calls, retrieval latency, peak GPU memory, GPU-seconds and wall time.
These budget values are starting conditions, not optimized hyperparameters.

## 10. Experimental baselines and ablation plan

All entries below are **[P] planned**, except the researcher-reported archived
conditions. No experiments were executed for this review.

| Condition | Purpose / controlled change |
|---|---|
| Archived zero-, 1-, 3-, 5-shot and QLoRA | Preserve exact original artifacts and protocol; no silent reruns. |
| Archived union, intersection, and train-calibrated source selection | Distinguish candidate availability from useful emission; no evidence advantage. Intersection/source-selection are new controls, not reported results. |
| Standard one-pass RAG | Same model, corpus and fixed retrieval budget; no verifier, completion or controller. |
| RAG + verification | Filter candidate set; no new-object discovery. |
| RAG + completion | Add evidence-extracted objects; no candidate-verification stage. |
| RAG + verification + fixed completion schedule | Same building blocks/actions as RAVEC; no adaptive control. |
| Fixed relation routing | Relation determines retrieve/verify/discover schedule; train-calibrated thresholds. CoRe-inspired, not falsely labelled a full CoRe reproduction. |
| Reproduced/adapted COVER-KBC, REAP and KGC-RAG | External adaptive, decomposition and retrieval controls; retain published parsing, information-access and budget differences in separate runs. |
| Adaptive RAVEC | Proposed action policy with same actions and maximum budget as fixed schedule. |
| RAVEC without constraints | Remove semantic constraint signals, retaining syntactic output schema and parser. |
| RAVEC without adaptive control | Fixed schedule with the same modules. |
| RAVEC without completion | Verify/accept/stop only; isolates recall recovery. |
| RAVEC without verification | Discovery without evidence acceptance checks; measures added-FP cost. |
| Uncertainty-only / complexity router / COVER-style heuristic | Tests whether separate action values add anything over established routing ideas. |
| Random legal routing / shuffled action values | Budget-matched negative controls, with fixed recorded seeds; no test-gold input. |
| No provenance grouping / duplicate-source injection | Measures whether copied corroboration creates false confidence. |
| Candidate-source and evidence-removal ablations | 5-shot only versus QLoRA only versus union; text evidence removed/shuffled; constraints as filters versus action triggers. |

### Metrics and statistical comparisons

**[V-local]** The pinned 2026 scorer averages **per-query** P, R and F1 across
rows; it is not an equal-weight mean over six relations. Macro F1 is not the
harmonic mean of macro P/R. Empty predictions have P=1, empty gold has R=1;
empty/empty F1=1. Micro metrics pool counts. Missing predictions become empty,
but the inference pipeline must still publish a complete ordered file. String
aliases use normalized one-to-one matching; numeric matching has a 5% relative
tolerance and known zero/negative/order edge cases. Preserve the scorer unchanged.
[Exact local definitions and provenance](evaluation.md#matching-and-metrics),
[vendored source](../src/llm_kbc/_vendor/lm_kbc_2026/evaluate.py)

**[P]** Report official macro and micro P/R/F1, overall and per relation, plus:
nonempty-gold performance; empty-gold false-positive rate and empty-case accuracy;
parse failures/truncations; candidate and final set sizes; matched TP/FP/FN;
newly recovered and accidentally removed TPs; evidence support precision/coverage;
retrieval gold-coverage diagnostics performed only *after* predictions are frozen;
runtime/cost and accepted TPs per retrieval/model call. Report an equal-relation
summary only as an explicitly additional metric, not the official macro score.

Use paired query-level comparisons with identical inputs. Bootstrap paired
differences with relation stratification; cluster repeated subject identities
across relations when applicable. Recompute official aggregates from each sample's
official per-row counts/scores rather than inventing a new matcher. Predeclare
the primary contrast and use multiplicity control for secondary claims. Report
95% intervals and effect sizes; ten award queries cannot support fine-grained
claims of small gains. If stochastic methods are reproduced, use several fixed
seeds and account separately for seed variance and query uncertainty.

Following the numeric caution in
[Coverage Is Not Knowledge](https://aclanthology.org/2026.akbc-1.33.pdf), compare
numeric pool coverage with relation- and magnitude-controlled shuffled-gold nulls
*offline only*. Such oracle/null analyses diagnose headroom, not deployable
verification. Do not run hidden-test diagnostic submissions to infer labels or
empty-answer priors.

### Development, calibration, held-out evaluation, and transfer

**[P]** Current validation is already development data. Do not rename it test or
claim a later train/validation split retrospectively makes it untouched. Use
cross-fitted training traces and reserve a disjoint calibration subset; remove
demonstration/self-answer leakage. Select a small number of hypotheses on current
validation, then freeze the complete procedure. Obtain a genuine organizer test
evaluation if still possible and permitted, or reserve untouched external
benchmark subjects/relations in advance. Record which labels were viewed and
when. Public test placeholders are never scoring gold.

For cross-model transfer, fit on pinned Qwen2.5-3B and freeze policy/thresholds
before applying to a verified second family. Separate zero-retuning from
target-training-only recalibration. Pin checkpoints, tokenizers, dtype, adapters,
chat templates and context/output budgets; do not change published baselines to
accommodate a new model. Different tokenizers make equal token counts only an
approximate cost match, so report wall time/FLOP or GPU-time proxies too.

## 11. Risks, limitations, and unresolved research questions

1. **[V/P] Prior-art saturation:** COVER-KBC already covers most of the proposed
   architecture; Self-RAG/Adaptive-RAG and KARMA cover major external-evidence
   ideas. A new acronym is not a contribution. The specific estimator and its
   controlled evidence must justify any novelty claim.
2. **[P] Information advantage:** external evidence, different model sizes and
   adapter training are confounds. Show matched comparisons and separate strata.
3. **[P] Open-world missingness:** insufficient evidence cannot establish an empty
   real-world answer, and a KB's gold set may omit true facts. Human adjudication
   of a blinded sample should be separate from immutable benchmark scoring.
4. **[V/P] Numeric chance:** wide guess pools gain accidental matches under
   tolerance; raw union TP gains alone do not prove complementary knowledge.
5. **[P] Temporal leakage:** newer webpages can reveal post-cutoff winners,
   listings or capacities. Snapshot policy and reproducible exclusions are needed.
6. **[P] Correlated verification:** same-model judgments and syndicated sources
   are not independent proof. Evidence entailment and factual truth differ.
7. **[P] Constraints:** types can be missing/wrong; cardinalities are not known
   from the subject alone. Shape conformance and ontology consistency are weaker
   than factual correctness; soft constraints can reduce recall.
8. **[V/P] Budget confounds:** 512-token arrays can truncate awards. Extra rounds,
   parsing retries or output allowance must be counted and exposed as factors.
9. **[U] Benchmark versions:** exact historical gold/scorer revisions and
   availability of final labels/services require pinning. Do not use paper model
   names as proof that a checkpoint works in the existing environment.
10. **[U] Paper/code discrepancies:** Relation-Specific Elicitation and CoRe
    contain inconsistent score/configuration descriptions; KAMEL releases disagree
    on split sizes/schema; Text2KGBench has inconsistent license wording.
11. **[P] Statistical power:** 475 rows, six relations and only ten awards do not
    justify a high-capacity controller or extensive hyperparameter search.
12. **[P] Reproducibility of retrieval:** live search changes rankings/documents.
    Cache complete provenance and lawful content snapshots; document unavailable
    pages and report query failures, not silently different evidence.

### Access limitations and exact materials still needed

**[U]** Primary paper PDFs were accessible for all five requested papers. Some
GitHub raw files/directories returned cache-miss errors; browser access to the
Self-RAG OpenReview page hit a verification challenge, but its official ICLR PDF
was accessible. These are access limitations, not proof that resources do not
exist. Request the following exact materials or fetch them in a later authorized
reproducibility audit before committing to the corresponding experiment:

| Material / exact source | Why still needed |
|---|---|
| [2023 scorer](https://github.com/lm-kbc/dataset2023/blob/main/evaluate.py), [2024 scorer](https://github.com/lm-kbc/dataset2024/blob/main/evaluate.py), and their pinned releases | Full executable scorer contents/revisions were not reliably accessible. Confirm ID/numeric/aggregation edge cases with parity fixtures before adapting. |
| [2023 test input](https://raw.githubusercontent.com/lm-kbc/dataset2023/main/data/test.jsonl), [2024 test input](https://raw.githubusercontent.com/lm-kbc/dataset2024/main/data/test.jsonl), [2025 test input](https://raw.githubusercontent.com/lm-kbc/dataset2025/master/data/test.jsonl) | These exact files were not retrievable through the browser. Establish label availability; do not infer it from filenames or repository publicity. |
| [KAMEL release](https://github.com/JanKalo/KAMEL/tree/master/data/KAMEL), [archive](https://github.com/JanKalo/KAMEL/blob/master/data/kamel.zip), KAMEL-specific license and official evaluator | Reconcile paper/script versus HF counts/schema, verify physical split sizes and label grouping, and clarify license attribution. No archive downloaded here. |
| [CoRe release](https://github.com/MightBeFonias/CoRe), [MAVE release](https://github.com/WSE-research/MAVE-LM-KBC-2026) | Both code links were inaccessible through this browser; require commit, license, prompts, scorer and runnable configuration. |
| [RARR release](https://github.com/anthonychen/RARR) | Paper advertises code, but direct repository retrieval failed; executable availability/license remains unverified. |
| [KGC-RAG release](https://github.com/jaejeajay/LM-KBC2024) | Repository landing page reachable; license, corpus reproduction and full executable configuration still need inspection. |
| [REAP repository](https://github.com/yammdd/AKBC-Shared-Task-2026), [Relation-Specific Elicitation repository](https://github.com/ruggsea/lmkbc2026-submission) license grants | Public visibility alone is not permission to copy code. The latter has no root license visible in the inspected listing; independently implementing a published method is a separate option. |
| [Coverage LICENSE](https://github.com/thylinao1/lm-kbc-2026/blob/main/LICENSE) and NOTICE | README declares Apache-2.0, but full license file retrieval failed; retain all notices before code reuse. |
| [COVER-KBC source/configuration](https://github.com/vquclinh/cover-kbc) at a fixed commit | README and paper accessible; full executable profile/code audit and parity run remain outstanding. Separate core caps from final relation escalations. |
| Archived thesis prediction/raw/metadata/evaluation files and hashes | Only researcher-supplied aggregate scores were available locally. Verify complete coverage, normalization, parser policy, model/adapter revisions and the union's 381/77 accounting. |
| Organizer clarification for Relation-Specific Elicitation's test result/configuration | Resolve PDF 0.6961 versus README 0.7121 and its absence from the final ranked table. Do not substitute either for an official final rank. |

## 12. Prioritized next research and engineering tasks

All tasks are **[P]** and require methodology approval before implementation.

1. **Freeze the research question and information-access contract.** Approve the
   verify-versus-discover hypothesis, precision margin, budgets, primary baseline,
   target-date/corpus rules and independent evaluation plan. Retain RAVEC as a
   working name only.
2. **First implementation task: an offline candidate/provenance audit.** Consume
   immutable 5-shot/QLoRA predictions plus raw/failure/metadata artifacts; validate
   hashes, unique identities and complete input coverage. Produce source-aware
   candidate differences and official-matcher accounting, including numeric
   chance controls and parse/truncation attribution. Gold may be used only in an
   explicitly offline development-analysis path, never in candidate construction
   or runtime controller signals. This establishes whether selection/recovery has
   measurable headroom before building RAG. No such code is implemented here.
3. Audit/pin COVER-KBC first and resolve external code licenses. Define exactly
   which reproduction is original versus a Qwen method adaptation. Do not build a
   generic plugin framework, multi-agent system or controller trainer.
4. Pin LM-KBC 2023 data/scorer/license and audit cross-edition identity/fact
   overlap. Reserve untouched transfer identities/relations. Resolve KAMEL release
   differences if broader replication is affordable.
5. Specify the smallest evidence snapshot and evidence-record schema, then build
   deterministic one-pass retrieval plus a standard RAG control. Verify timestamp,
   provenance, failure handling and artifact integrity before GPU experiments.
6. Add evidence verification and a fixed recovery schedule; establish their
   separate error/cost effects. If they cannot jointly help, do not add a learned
   controller to hide a weak pipeline.
7. Collect bounded train-only action traces with cross-fitting. Compare a simple
   value policy against static schedules and COVER-style heuristics, keeping the
   action space identical.
8. Run the preregistered paired ablations and transfer study. Archive full
   precision output, official scorer provenance, corpus/index/model hashes,
   prompts, action traces, costs, Slurm environment/job IDs and random seeds.
   Do not select final claims from the best validation run alone.

## 13. References with verified links

The links below were accessible unless specifically identified as unavailable in
§11. Publisher PDFs take precedence over inconsistent HTML abstracts; live code
repositories are resources, not immutable proof of a paper's submitted version.

1. Vo (2026). [COVER-KBC](https://aclanthology.org/2026.akbc-1.37/).
   [PDF](https://aclanthology.org/2026.akbc-1.37.pdf), [code](https://github.com/vquclinh/cover-kbc).
2. Bui, Do and Nguyen (2026). [REAP](https://aclanthology.org/2026.akbc-1.31/).
   [PDF](https://aclanthology.org/2026.akbc-1.31.pdf), [code](https://github.com/yammdd/AKBC-Shared-Task-2026).
3. Lazzaroni et al. (2026). [Relation-Specific Elicitation](https://aclanthology.org/2026.akbc-1.45/).
   [Authoritative PDF](https://aclanthology.org/2026.akbc-1.45.pdf), [live code](https://github.com/ruggsea/lmkbc2026-submission).
4. Habibillah and Tjiaranata (2026). [CoRe](https://aclanthology.org/2026.akbc-1.42/).
   [PDF](https://aclanthology.org/2026.akbc-1.42.pdf).
5. Silchenko (2026). [Coverage Is Not Knowledge](https://aclanthology.org/2026.akbc-1.33/).
   [PDF](https://aclanthology.org/2026.akbc-1.33.pdf), [code](https://github.com/thylinao1/lm-kbc-2026).
6. Kunze and Both (2026). [MAVE](https://aclanthology.org/2026.akbc-1.38/).
   [PDF](https://aclanthology.org/2026.akbc-1.38.pdf).
7. Kalo et al. (2026). [Shared-task overview](https://aclanthology.org/2026.akbc-1.28.pdf).
   [Rules](https://www.akbc.ws/2026/shared-task.html), [final ranked results](https://www.akbc.ws/2026/AKBC%20Shared%20Task%202026%20Final%20Leaderboard.pdf).
8. Asai et al. (ICLR 2024). [Self-RAG, official proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf).
   [Author resources](https://selfrag.github.io/).
9. Jiang et al. (EMNLP 2023). [Active Retrieval Augmented Generation / FLARE](https://aclanthology.org/2023.emnlp-main.495/).
10. Jeong et al. (NAACL 2024). [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/).
11. Gao et al. (ACL 2023). [RARR](https://aclanthology.org/2023.acl-long.910/).
12. Dhuliawala et al. (Findings ACL 2024). [Chain-of-Verification](https://aclanthology.org/2024.findings-acl.212/).
13. Wei et al. (Findings EMNLP 2023). [KICGPT](https://aclanthology.org/2023.findings-emnlp.580/).
14. Singhania, Razniewski and Weikum (RepL4NLP 2023).
    [Extracting Multi-valued Relations](https://aclanthology.org/2023.repl4nlp-1.12/).
15. Albert-Roulhac and Zouaq (2025). [ReWiSe](https://ceur-ws.org/Vol-4041/paper8.pdf).
16. He and Razniewski (2025). [Self-Retrieval + Divide-and-Conquer](https://ceur-ws.org/Vol-4041/paper11.pdf).
17. Lu et al. (NeurIPS 2025). [KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment](https://proceedings.neurips.cc/paper_files/paper/2025/file/517f9b9c227b9dd51dba4560f37165ed-Paper-Conference.pdf).
    Do not confuse it with the different 2026 paper also named KARMA.
18. Jiang et al. (IEEE ICSC 2025). [OntoSCPrompt](https://www.inf.uni-hamburg.de/en/inst/ab/sems/resources/jiang-et-al-icsc-2025-ontology.pdf).
19. Mihindukulasooriya et al. (ISWC 2023). [Text2KGBench author manuscript](https://arxiv.org/html/2308.02357),
    [dataset release](https://zenodo.org/records/7916716).
20. Gao et al. (EMNLP 2023). [ALCE: Enabling LLMs to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/).
21. Kalo et al. (2023). [LM-KBC 2023 overview](https://ceur-ws.org/Vol-3577/paper0.pdf),
    [official site](https://lm-kbc.github.io/challenge2023/), [dataset](https://github.com/lm-kbc/dataset2023).
22. Kalo et al. (2024). [LM-KBC 2024 overview](https://ceur-ws.org/Vol-3853/paper0.pdf),
    [official site/results](https://lm-kbc.github.io/challenge2024/), [dataset](https://github.com/lm-kbc/dataset2024).
23. [LM-KBC 2025 official site/results](https://lm-kbc.github.io/challenge2025/),
    [dataset](https://github.com/lm-kbc/dataset2025), [scorer](https://raw.githubusercontent.com/lm-kbc/dataset2025/master/evaluate.py).
24. Kalo and Fichtel (AKBC 2022). [KAMEL paper](https://akbc.ws/2022/assets/pdfs/15_kamel_knowledge_analysis_with_.pdf),
    [repository](https://github.com/JanKalo/KAMEL), [author dataset card](https://huggingface.co/datasets/LeandraFichtel/KAMEL),
    [split-generation source](https://raw.githubusercontent.com/JanKalo/KAMEL/master/create_train_dev_test_split.py).
25. Mesquita et al. (EMNLP-IJCNLP 2019). [KnowledgeNet](https://aclanthology.org/D19-1069/),
    [official repository/evaluation](https://github.com/diffbot/knowledge-net).
26. [Official Mistral-Small-3.2-24B card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
    and [Gemma-4-31B card](https://huggingface.co/google/gemma-4-31B).
27. Prabhong, Kertkeidkachorn and Trongratsameethong (LM-KBC 2024).
    [KGC-RAG](https://ceur-ws.org/Vol-3853/paper13.pdf),
    [code](https://github.com/jaejeajay/LM-KBC2024).
28. Bara (LM-KBC 2024). [Prompt Engineering for Tail Prediction in Domain-Specific Knowledge Graph Completion Tasks](https://ceur-ws.org/Vol-3853/paper10.pdf).

No new benchmark score, training run, retrieval experiment, or novelty claim has
been established by this document.
