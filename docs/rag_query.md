# RAG-04.2: standalone query construction

`llm_kbc.rag_query` is a deterministic, standard-library-only query boundary.
It supports arbitrary subjects and relation identifiers independently of models,
dataset readers, Wikipedia, retrieval, prompts and evaluation. This establishes
input/interface support, **not demonstrated end-to-end generalization** to another
KBC dataset. No existing pipeline imports or uses this module yet.

## Interface and validation

```python
from llm_kbc.rag_query import QueryInput, build_query, validate_query_input

task = validate_query_input({
    "subject": "Fictional Observatory",
    "relation_id": "locatedIn",
    "relation_description": "city where the observatory is located",
    "relation_namespace": "synthetic-kbc",
})
result = build_query(task)
assert result["query"] == "Fictional Observatory city where the observatory is located"
```

`QueryInput` is a frozen dataclass with required `subject` and `relation_id`
strings, plus optional `relation_description` and `relation_namespace` strings.
`validate_query_input()` accepts the dataclass or a mapping, returns a validated
immutable instance, and performs no query construction or relation-whitelist
check. `build_query()` always revalidates its input, then returns a fresh provenance
dictionary. Direct dataclass construction alone does not validate runtime types.

Required strings must be nonblank; optional strings may be absent/`None`, but not
blank. All strings must be valid UTF-8-encodable Unicode without NUL. Unknown
mapping fields are rejected, including `ObjectEntities`, gold and predictions.
Dataset adapters must explicitly project only the four permitted fields; they
must not derive descriptions or namespaces from object labels or model answers.
No dataset loader, file access, network access or model dependency is involved.

## Fixed policy: `rag-query-v1`

Precedence is:

1. Only the exact namespace **`lm-kbc-2026`** and an exact recognized relation ID
   activate the six existing `subject-relation-lexical-v1` templates below.
2. Otherwise, an explicit description produces `{subject} {relation_description}`.
3. Otherwise, the literal fallback produces `{subject} {relation_id}`.

| LM-KBC relation | Exact template |
|---|---|
| `awardWonBy` | `{subject} award winners recipients` |
| `companyTradesAtStockExchange` | `{subject} stock exchange listing` |
| `countryLandBordersCountry` | `{subject} land borders neighbouring countries` |
| `hasArea` | `{subject} total area` |
| `hasCapacity` | `{subject} capacity` |
| `personHasCityOfDeath` | `{subject} place of death` |

Known LM-KBC templates take precedence over a supplied description, which remains
recorded and fingerprinted. The caller is responsible for choosing a truthful
namespace and compatible definition. No namespace inference, aliases or spelling
correction occurs. An absent namespace, another dataset namespace, or differently
cased namespace cannot activate LM-KBC templates even if its relation ID matches.
Unknown relations within the LM-KBC namespace still use the generic precedence.
Compatibility templates are intentionally isolated from unchanged BM25 v1 code;
parity tests compare all six directly to the existing builder.

Examples (synthetic inputs):

| Input | Query | Method |
|---|---|---|
| `Synthetic object`, `hasArea`, namespace `lm-kbc-2026` | `Synthetic object total area` | `lm_kbc_2026_template` |
| `Synthetic object`, `hasArea`, namespace `another-dataset`, description `administrative district` | `Synthetic object administrative district` | `relation_description` |
| `Fictional Observatory`, `locatedIn`, no description | `Fictional Observatory locatedIn` | `literal_relation_fallback` |
| `Fictional Observatory`, `P999999`, no description | `Fictional Observatory P999999` | `literal_relation_fallback` |

Every literal fallback has `limited_fallback: true`. Opaque identifiers are not
decoded or classified: readable and opaque identifiers both follow the same
mechanical rule. The module does not infer direction, units, answer types, entity
identity or expected answers. Descriptions are caller-supplied definitions, not
independently verified semantics. Arbitrary namespace strings are supported;
namespace governance across datasets remains the adapter's responsibility.

## Provenance and canonical fingerprints

Every result contains the four input fields (absent optionals become `null`),
`query_policy_version`, `method`, `template_id` (namespace/ID for a recognized
LM-KBC template, otherwise `null`), the exact `template`, `limited_fallback`,
constructed `query`, and three SHA256 hex fingerprints:

- `identity_sha256`: subject, relation ID and namespace; excludes description.
- `task_definition_sha256`: relation ID, namespace and description; excludes subject.
- `query_sha256`: every other provenance field, including the two fingerprints.

For each fingerprint, add a `fingerprint_type` field to its payload, with value
`rag-query-identity-v1`, `rag-query-task-definition-v1` or
`rag-query-provenance-v1`, respectively. Serialize using
`json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
allow_nan=False)`, append exactly one LF, encode UTF-8, then compute SHA256.
`query_sha256` is a provenance fingerprint, **not a hash of query text alone**.

No stripping, whitespace collapse, case folding, Unicode normalization, numeric
conversion or punctuation rewriting occurs. Generic templates insert one ASCII
space between the two supplied strings; whitespace inside either string remains
unchanged. Embedded newlines/tabs are preserved and JSON-escaped. Optional fields
omitted versus explicitly `None` produce identical provenance. Mapping insertion
order has no effect. Canonically equivalent Unicode spellings are distinct inputs.
These hashes establish reproducible identity/integrity, not factual correctness.

A description change preserves the identity hash but changes the task and query
hashes, even when an LM-KBC template keeps the rendered query unchanged. A subject
change preserves the task hash; namespace/ID changes affect identity and task
hashes even if the rendered query text is identical. Inputs are never mutated.

## Integration boundary and limitations

Existing discovery v1, BM25, identity loading, pilot SHA checks, historical
collection, preprocessing, baseline prompts, inference and evaluators remain
unchanged. This module does not migrate artifact schemas, change article limits,
or reinterpret archived discovery fingerprints as these new fingerprints.
RAG-04.3 will explicitly integrate generic queries into versioned discovery.
Archive a module/code revision alongside future artifacts; this stage does not
publish files or define an artifact completion protocol.

No corpus expansion, retrieval ranking or generation is implemented here. Query
quality for new relations/datasets is unmeasured; literal identifiers may be
uninformative, and user-supplied descriptions may be ambiguous or incorrect.
Whitespace and Wikimedia search operators are not escaped for a specific search
engine. Any future backend-specific encoding must have an explicit versioned
policy. Unsupported descriptions must never be repaired using benchmark answers.
