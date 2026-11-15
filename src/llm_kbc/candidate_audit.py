"""Offline, gold-aware candidate audit; never a runtime selection policy.

All scores and TP counterfactuals call the unchanged official evaluator. The
stable union retains five-shot candidates first, then unseen exact QLoRA strings
in original order. No candidate is generated from gold labels.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import tempfile
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path

from .dataset_inspector import DatasetError, load_jsonl
from .prediction_io import load_predictions
from .zero_shot import InferenceError, _parse_response


POLICY_VERSION = "ravec-00-candidate-audit-v1"


class AuditError(ValueError):
    """Invalid, incomplete, or inconsistent audit inputs."""


def _identity(row):
    return row["SubjectEntity"], row["Relation"]


def _identities(rows, name):
    identities = [_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise AuditError(f"{name}: duplicate subject-relation identities")
    return identities


def _fingerprint(path):
    content = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _json_rows(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"{path}:{index}: malformed JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise AuditError(f"{path}:{index}: expected an object")
        rows.append(row)
    return rows


def _artifact_audit(predictions, paths, provenance, gold_hash):
    """Verify optional sidecars without inferring absent diagnostics as zero.

    Raw logs may be a prefix (e.g. an interrupted or incompletely archived log).
    Completeness is reported explicitly; available rows must agree with parsing
    and predictions. Existing logs have no generated-token/finish-reason fields.
    """
    diagnostics = {"raw": None, "failures": None, "metadata": None,
                   "truncation": "Unknown: existing logs do not record token counts or finish reasons"}
    identities = [_identity(row) for row in predictions]
    raw_failures = []
    for kind, path in paths.items():
        if path is None:
            continue
        provenance[kind] = _fingerprint(path)
        if kind == "metadata":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise AuditError(f"{path}: expected metadata object")
            if not isinstance(metadata.get("dataset", {}), dict):
                raise AuditError(f"{path}: dataset metadata must be an object")
            for key in ("total_inputs", "processed_records"):
                if key in metadata and (type(metadata[key]) is not int or metadata[key] != len(predictions)):
                    raise AuditError(f"{path}: {key} differs from complete prediction coverage")
            for actual, expected, label in (
                (metadata.get("output_sha256"), provenance["predictions"]["sha256"], "prediction"),
                (metadata.get("dataset", {}).get("sha256"), gold_hash, "dataset"),
            ):
                if actual is not None and actual != expected:
                    raise AuditError(f"{path}: {label} SHA256 mismatch")
            diagnostics[kind] = {key: metadata.get(key) for key in (
                "status", "total_inputs", "processed_records", "valid_records", "parse_failures",
                "format_validity_rate", "numeric_normalized_responses", "parse_failure_policy",
                "model_id", "requested_revision", "resolved_revision", "prompt_version", "prompt_sha256",
                "source_sha256", "adapter", "manifest", "chat_template_sha256", "generation", "versions",
            )}
            diagnostics[kind]["prediction_hash_verified"] = metadata.get("output_sha256") is not None
            diagnostics[kind]["dataset_hash_verified"] = metadata.get("dataset", {}).get("sha256") is not None
            continue
        rows = _json_rows(path)
        seen = set()
        counts = Counter()
        by_relation = defaultdict(Counter)
        for position, row in enumerate(rows, 1):
            index = row.get("input_line")
            if type(index) is not int or not 1 <= index <= len(identities) or index in seen:
                raise AuditError(f"{path}:{position}: invalid or duplicate input_line")
            if seen and index <= max(seen):
                raise AuditError(f"{path}:{position}: input order differs")
            seen.add(index)
            if (row.get("SubjectEntity"), row.get("Relation")) != identities[index - 1]:
                raise AuditError(f"{path}:{position}: identity differs from predictions")
            response = row.get("response")
            if not isinstance(response, str):
                raise AuditError(f"{path}:{position}: missing raw response")
            try:
                objects, normalized = _parse_response(response)
                status = "valid"
            except InferenceError:
                objects, normalized, status = [], False, "parse_failed"
            if status == "parse_failed" and not isinstance(row.get("error"), str):
                raise AuditError(f"{path}:{position}: parse failure is missing its error")
            if kind == "raw":
                if index != position or row.get("parsing_status") != status:
                    raise AuditError(f"{path}:{position}: raw order/parsing status differs")
                if objects != predictions[index - 1]["ObjectEntities"]:
                    raise AuditError(f"{path}:{position}: parsed response differs from predictions")
            elif status != "parse_failed" or predictions[index - 1]["ObjectEntities"] != []:
                raise AuditError(f"{path}:{position}: invalid parse-failure abstention")
            counts[status] += 1
            by_relation[row["Relation"]][status] += 1
            counts["numeric_normalized"] += normalized
            # Lexical diagnostic only: never a claim of token-budget exhaustion.
            if status == "parse_failed":
                counts["failed_array_without_closing_bracket"] += (
                    response.strip().startswith("[") and not response.strip().endswith("]")
                )
                if kind == "raw":
                    raw_failures.append((index, response))
        diagnostics[kind] = {"records": len(rows), "counts": dict(sorted(counts.items())),
                             "per_relation": {r: dict(c) for r, c in sorted(by_relation.items())},
                             "complete": len(rows) == len(predictions) if kind == "raw" else None}
        if kind == "failures" and paths.get("raw") is not None:
            expected = [(row["input_line"], row["response"]) for row in rows]
            if expected != raw_failures:
                raise AuditError(f"{path}: failure log differs from available raw log")
    if diagnostics["raw"] is not None and diagnostics["raw"]["complete"] and diagnostics["metadata"] is not None:
        counts = diagnostics["raw"]["counts"]
        for field, count in (("valid_records", counts.get("valid", 0)),
                             ("parse_failures", counts.get("parse_failed", 0)),
                             ("numeric_normalized_responses", counts.get("numeric_normalized", 0))):
            reported = diagnostics["metadata"][field]
            if reported is not None and reported != count:
                raise AuditError(f"Metadata {field} differs from complete raw log")
    return diagnostics


def _score(official, objects, gold):
    prediction = {**gold, "ObjectEntities": objects}
    return official.evaluate_per_sr_pair([prediction], [gold], official.RELATION_TYPE, 0.05)[0]


def _metrics(official, scores):
    # Match upstream's row order even for floating-point summation at full precision.
    scores = sorted(scores, key=lambda row: (row["Relation"], row["SubjectEntity"]))
    macro = official.macro_average_per_relation(scores)
    micro = official.micro_average_per_relation(scores)
    return {key: {**macro[key], **micro[key]} for key in macro}


def _distribution(values):
    ordered = sorted(values)
    return {"mean": sum(values) / len(values), "min": ordered[0], "max": ordered[-1],
            "p025": ordered[math.floor((len(values) - 1) * 0.025)],
            "p975": ordered[math.ceil((len(values) - 1) * 0.975)]}


def numeric_null_controls(official, gold, five, union, *, iterations, seed):
    """Permute positive singleton numeric gold groups, keeping pools fixed.

    Controls are diagnostic, not evidence of factual knowledge or calibrated
    hypothesis tests. Magnitude bins use floor(log10(first gold alias)). Rows
    with zero/negative/nonfinite/unparseable or nonsingleton gold are excluded
    only from these controls, never from the actual official evaluation.
    """
    result = {}
    for relation, relation_type in sorted(official.RELATION_TYPE.items()):
        if relation_type != "numeric":
            continue
        indices = [i for i, row in enumerate(gold) if row["Relation"] == relation]
        eligible, magnitudes = [], {}
        for i in indices:
            groups = gold[i]["ObjectEntities"]
            value = official.try_parse_number(groups[0][0]) if len(groups) == 1 else None
            if value is not None and math.isfinite(value) and value > 0:
                eligible.append(i)
                magnitudes[i] = math.floor(math.log10(value))
        relation_result = {"eligible_queries": len(eligible), "excluded_queries": len(indices) - len(eligible)}
        for mode in ("within_relation", "within_relation_magnitude_decade"):
            buckets = defaultdict(list)
            for i in eligible:
                buckets[magnitudes[i] if mode.endswith("decade") else 0].append(i)
            observed = {name: sum(_score(official, rows[i]["ObjectEntities"], gold[i])["tp"]
                                  for i in eligible) for name, rows in (("five_shot", five), ("union", union))}
            observed["tp_gain"] = observed["union"] - observed["five_shot"]
            samples = {key: [] for key in observed}
            fixed = 0
            rng = random.Random(seed)
            for _ in range(iterations):
                totals = {"five_shot": 0, "union": 0}
                for bucket in buckets.values():
                    donors = bucket.copy()
                    rng.shuffle(donors)  # Fixed points intentionally retained and quantified.
                    fixed += sum(target == donor for target, donor in zip(bucket, donors))
                    for target, donor in zip(bucket, donors):
                        shuffled_gold = {**gold[target], "ObjectEntities": gold[donor]["ObjectEntities"]}
                        for name, rows in (("five_shot", five), ("union", union)):
                            totals[name] += _score(official, rows[target]["ObjectEntities"], shuffled_gold)["tp"]
                totals["tp_gain"] = totals["union"] - totals["five_shot"]
                for key in samples:
                    samples[key].append(totals[key])
            relation_result[mode] = {
                "observed": observed, "null": {key: _distribution(values) for key, values in samples.items()},
                "tp_gain_upper_tail_fraction_plus_one": (
                    1 + sum(value >= observed["tp_gain"] for value in samples["tp_gain"])
                ) / (iterations + 1),
                "bucket_sizes": {str(key): len(bucket) for key, bucket in sorted(buckets.items())},
                "movable_queries": sum(len(bucket) for bucket in buckets.values() if len(bucket) > 1),
                "fixed_identity_fraction": fixed / (iterations * len(eligible)) if eligible else None,
                "available": bool(eligible),
            }
        result[relation] = relation_result
    return result


def audit_candidates(*, gold_path, five_shot_path, qlora_path, sidecars=None,
                     union_path=None, iterations=1000, seed=0, expected=None):
    """Audit complete ordered validation predictions without changing artifacts.

    sidecars maps 'five_shot'/'qlora' to explicit optional raw/failures/metadata
    paths. Missing inputs remain None; no implicit archive-path assumptions.
    """
    if type(iterations) is not int or iterations < 1 or type(seed) is not int:
        raise AuditError("iterations must be positive and seed must be an integer")
    gold_path, five_shot_path, qlora_path = map(Path, (gold_path, five_shot_path, qlora_path))
    if gold_path.resolve().name != "val.jsonl":
        raise AuditError("Explicit validation gold must be named val.jsonl; never use public test labels")
    test_path = gold_path.parent / "test.jsonl"
    if test_path.exists() and gold_path.samefile(test_path):
        raise AuditError("Validation gold refers to public test.jsonl")
    # pandas is already the project's optional evaluation dependency.
    from ._vendor.lm_kbc_2026 import evaluate as official

    root = files("llm_kbc._vendor.lm_kbc_2026")
    upstream = json.loads(root.joinpath("provenance.json").read_text(encoding="utf-8"))
    for name, entry in upstream["files"].items():
        if hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() != entry["sha256"]:
            raise AuditError(f"Vendored {name} differs from its provenance")
    provenance = {"gold": _fingerprint(gold_path)}
    gold = list(load_jsonl(gold_path))
    identities = _identities(gold, "gold")
    if not gold:
        raise AuditError("Empty validation gold")
    if any(row["Relation"] not in official.RELATION_TYPE for row in gold):
        raise AuditError("Unsupported relation in gold")
    predictions, diagnostics = {}, {}
    sidecars = sidecars or {}
    for name, path in (("five_shot", five_shot_path), ("qlora", qlora_path)):
        provenance[name] = {"predictions": _fingerprint(path)}
        rows = list(load_predictions(path))
        if _identities(rows, name) != identities:
            raise AuditError(f"{name}: coverage or input order differs from validation gold")
        predictions[name] = rows
        paths = {kind: Path(value) if value is not None else None
                 for kind, value in sidecars.get(name, {}).items()}
        if set(paths) - {"raw", "failures", "metadata"}:
            raise AuditError(f"{name}: unknown sidecar kind")
        paths = {kind: paths.get(kind) for kind in ("raw", "failures", "metadata")}
        diagnostics[name] = _artifact_audit(rows, paths, provenance[name], provenance["gold"]["sha256"])

    query_reports, union, scores = [], [], {name: [] for name in ("five_shot", "qlora", "union")}
    relation_counts = defaultdict(Counter)
    for index, (g, f, q) in enumerate(zip(gold, predictions["five_shot"], predictions["qlora"]), 1):
        five, qlora = f["ObjectEntities"], q["ObjectEntities"]
        raw_additions = [value for value in dict.fromkeys(qlora) if value not in set(five)]
        combined = list(dict.fromkeys(five + qlora))
        union.append({**f, "ObjectEntities": combined})
        normalized_five = {official.normalize_string(value) for value in five}
        retained, candidates = five.copy(), []
        previous_tp = _score(official, retained, g)["tp"]
        seen = normalized_five.copy()
        for value in raw_additions:
            normalized = official.normalize_string(value)
            collision = normalized in seen
            singleton_tp = _score(official, [value], g)["tp"]
            retained.append(value)
            current_tp = _score(official, retained, g)["tp"]
            candidates.append({"value": value, "normalized": normalized,
                               "normalized_collision": collision, "individually_matchable": bool(singleton_tp),
                               "prefix_marginal_tp": current_tp - previous_tp})
            previous_tp = current_tp
            seen.add(normalized)
        local_scores = {name: _score(official, values, g) for name, values in (
            ("five_shot", five), ("qlora", qlora), ("union", combined))}
        counts = Counter({
            "queries": 1, "raw_additional_unique_strings": len(raw_additions),
            "raw_additional_occurrences": sum(value not in set(five) for value in qlora),
            "normalized_additional_candidates": len(seen - normalized_five),
            "tp_gain": local_scores["union"]["tp"] - local_scores["five_shot"]["tp"],
            "added_individually_unmatched": sum(not c["individually_matchable"] for c in candidates),
            "added_matchable_without_prefix_gain": sum(c["individually_matchable"] and not c["prefix_marginal_tp"] for c in candidates),
        })
        for name, values in (("five_shot", five), ("qlora", qlora)):
            counts[f"{name}_empty_queries"] = not values
            counts[f"{name}_exact_duplicate_occurrences"] = len(values) - len(set(values))
            counts[f"{name}_normalized_duplicate_occurrences"] = len(values) - len({official.normalize_string(v) for v in values})
        for name, score in local_scores.items():
            scores[name].append(score)
            counts[f"{name}_tp"] = score["tp"]
            counts[f"{name}_fp"] = score["total_pred"] - score["tp"]
            counts[f"{name}_fn"] = score["total_gt"] - score["tp"]
        relation_counts[g["Relation"]].update(counts)
        query_reports.append({"input_line": index, "SubjectEntity": g["SubjectEntity"],
                              "Relation": g["Relation"], "counts": dict(counts),
                              "scores": local_scores, "qlora_raw_additions": candidates})
    if union_path is not None:
        union_path = Path(union_path)
        provenance["archived_union"] = _fingerprint(union_path)
        archived = list(load_predictions(union_path))
        if _identities(archived, "archived union") != identities:
            raise AuditError("Archived union coverage/order differs")
        union_comparison = {
            "exact_objects_and_order_match": all(a["ObjectEntities"] == b["ObjectEntities"] for a, b in zip(archived, union)),
        }
        # Official rows are sorted by relation/subject, unlike the audit's input order.
        union_comparison["official_scores_match"] = (
            official.evaluate_per_sr_pair(archived, gold, official.RELATION_TYPE, 0.05)
            == sorted(scores["union"], key=lambda row: (row["Relation"], row["SubjectEntity"]))
        )
    else:
        union_comparison = None
    totals = Counter()
    for counts in relation_counts.values():
        totals.update(counts)
    expected = expected or {}
    if set(expected) - {"queries", "raw_additional_unique_strings", "tp_gain"}:
        raise AuditError("Unknown reference accounting key")
    checks = {key: {"expected": value, "observed": totals[key], "matches": totals[key] == value}
              for key, value in sorted(expected.items())}
    report = {
        "policy_version": POLICY_VERSION, "status": "complete", "offline_gold_analysis": True,
        "configuration": {"null_iterations": iterations, "seed": seed, "tolerance": 0.05,
                          "union_order": "five-shot first; unseen exact QLoRA strings in file order"},
        "provenance": provenance, "upstream": upstream,
        "software": {"python": platform.python_version(), "pandas": official.pd.__version__,
                     "audit_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "totals": dict(totals), "per_relation": {r: dict(c) for r, c in sorted(relation_counts.items())},
        "metrics": {name: _metrics(official, rows) for name, rows in scores.items()},
        "reference_checks": checks, "archived_union_comparison": union_comparison,
        "artifact_diagnostics": diagnostics, "queries": query_reports,
        "numeric_null_controls": numeric_null_controls(
            official, gold, predictions["five_shot"], union, iterations=iterations, seed=seed),
    }
    # Detect ordinary concurrent input changes before publishing any audit output.
    for group in provenance.values():
        entries = [group] if "sha256" in group else group.values()
        for entry in entries:
            if _fingerprint(Path(entry["path"])) != entry:
                raise AuditError(f"Input changed during audit: {entry['path']}")
    report["summary"] = summarize(report)
    return report


def summarize(report):
    """Concise findings plus limits; no novelty or candidate-precision claims."""
    totals = report["totals"]
    lines = ["# RAVEC-00 offline candidate audit", "",
             f"Queries: {totals['queries']}; additional unique raw QLoRA strings: "
             f"{totals['raw_additional_unique_strings']}; normalized additions: "
             f"{totals['normalized_additional_candidates']}; official net TP gain: {totals['tp_gain']}.", "",
             "| Relation | Raw additions | Normalized additions | TP gain | Five-shot FP | Union FP |",
             "|---|---:|---:|---:|---:|---:|"]
    for relation, counts in report["per_relation"].items():
        lines.append(f"| {relation} | {counts['raw_additional_unique_strings']} | "
                     f"{counts['normalized_additional_candidates']} | {counts['tp_gain']} | "
                     f"{counts['five_shot_fp']} | {counts['union_fp']} |")
    table = "\n".join(lines)
    lines = [table, "Reference checks: " + (json.dumps(report["reference_checks"], sort_keys=True)
               if report["reference_checks"] else "not supplied"),
              "TP gain / raw additions is not candidate precision: aliases, normalized collisions, "
              "one-to-one matching and greedy numeric order affect credit. Prefix marginal credit is order-dependent.",
              "Verify-versus-discover: prioritize verification where unmatched additions/FP grow; "
              "consider discovery only where relation-specific gains survive numeric null diagnostics. "
              "These are offline headroom signals, not runtime features or proof of factual knowledge."]
    for name, diagnostics in report["artifact_diagnostics"].items():
        lines.append(f"{name} sidecars: " + ", ".join(
            f"{kind}={'missing' if diagnostics[kind] is None else 'available'}"
            for kind in ("raw", "failures", "metadata")) + ". " + diagnostics["truncation"] + ".")
    for relation, control in report["numeric_null_controls"].items():
        for mode in ("within_relation", "within_relation_magnitude_decade"):
            values = control[mode]
            lines.append(f"{relation} / {mode}: eligible={control['eligible_queries']}, "
                         f"movable={values['movable_queries']}, observed TP gain={values['observed']['tp_gain']}, "
                         f"null mean={values['null']['tp_gain']['mean']:.4f}, "
                         f"null 95% range=[{values['null']['tp_gain']['p025']}, {values['null']['tp_gain']['p975']}].")
    lines.append("Nulls are descriptive shuffled-gold controls, with fixed points and singleton bins; "
                 "not calibrated significance tests. Validation has already been inspected. No RAVEC superiority claim.")
    return "\n\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Explicit labeled val.jsonl")
    parser.add_argument("--five-shot", type=Path, required=True)
    parser.add_argument("--qlora", type=Path, required=True)
    parser.add_argument("--union", type=Path, help="Optional archived union to compare")
    parser.add_argument("--output", type=Path, required=True, help="New JSON audit; never overwritten")
    parser.add_argument("--null-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    for name in ("five-shot", "qlora"):
        for kind in ("raw", "failures", "metadata"):
            parser.add_argument(f"--{name}-{kind}", type=Path)
    for option in ("queries", "raw-additions", "tp-gain"):
        parser.add_argument(f"--expected-{option}", type=int)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise AuditError(f"Refusing to overwrite {args.output}")
        if not args.output.parent.is_dir():
            raise AuditError("Output parent must already exist")
        expected = {key: value for key, value in (
            ("queries", args.expected_queries), ("raw_additional_unique_strings", args.expected_raw_additions),
            ("tp_gain", args.expected_tp_gain)) if value is not None}
        report = audit_candidates(
            gold_path=args.gold, five_shot_path=args.five_shot, qlora_path=args.qlora,
            union_path=args.union, iterations=args.null_iterations, seed=args.seed, expected=expected,
            sidecars={name: {kind: getattr(args, f"{name}_{kind}") for kind in ("raw", "failures", "metadata")}
                      for name in ("five_shot", "qlora")},
        )
        # One complete report, published no-clobber only after all validation.
        # Same-directory hard links require a compatible local/HPC filesystem.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(report["summary"], end="")
        return 0 if all(check["matches"] for check in report["reference_checks"].values()) else 1
    except (AuditError, DatasetError, OSError, UnicodeError, json.JSONDecodeError, ModuleNotFoundError) as error:
        parser.exit(2, f"Audit failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
