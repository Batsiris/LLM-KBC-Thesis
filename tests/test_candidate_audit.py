"""Synthetic audit fixtures; these are not thesis experiment results."""

import hashlib
import json

import pytest

from llm_kbc import candidate_audit as audit
from llm_kbc._vendor.lm_kbc_2026 import evaluate as official
from llm_kbc.dataset_inspector import DatasetError


RELATION = "countryLandBordersCountry"


def row(objects, subject="A", relation=RELATION):
    return {"SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects}


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def inputs(tmp_path, gold=None, five=None, qlora=None):
    return {
        "gold_path": write(tmp_path / "val.jsonl", gold if gold is not None else [row([["France", "FR"]])]),
        "five_shot_path": write(tmp_path / "five.jsonl", five if five is not None else [row(["FR", "FR"])]),
        "qlora_path": write(tmp_path / "qlora.jsonl", qlora if qlora is not None else [row(["fr", "France", "Spain", "Spain"])]),
        "iterations": 7,
    }


def test_aliases_duplicates_unmatched_and_attribution(tmp_path):
    kwargs = inputs(tmp_path)
    before = {p: p.read_bytes() for p in kwargs.values() if hasattr(p, "read_bytes")}
    result = audit.audit_candidates(**kwargs)
    counts = result["totals"]
    assert counts["raw_additional_unique_strings"] == 3
    assert counts["raw_additional_occurrences"] == 4
    assert counts["normalized_additional_candidates"] == 2
    assert counts["tp_gain"] == 0
    assert counts["added_individually_unmatched"] == 1
    assert counts["added_matchable_without_prefix_gain"] == 2
    assert counts["five_shot_exact_duplicate_occurrences"] == 1
    assert counts["qlora_normalized_duplicate_occurrences"] == 1
    assert counts["union_fp"] == 2  # France is another alias, but not a second gold object.
    additions = result["queries"][0]["qlora_raw_additions"]
    assert [c["value"] for c in additions] == ["fr", "France", "Spain"]
    assert additions[0]["normalized_collision"] is True
    assert additions[-1]["individually_matchable"] is False
    assert result == audit.audit_candidates(**kwargs)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert result["artifact_diagnostics"]["five_shot"]["raw"] is None


def test_numeric_tolerance_and_official_parity(tmp_path):
    gold = [row([["100"]], "A", "hasArea"), row([["100"]], "B", "hasCapacity"), row([], "C")]
    five = [row(["94.99"], "A", "hasArea"), row([], "B", "hasCapacity"), row([], "C")]
    qlora = [row(["105", "105.01"], "A", "hasArea"), row(["95"], "B", "hasCapacity"), row([], "C")]
    result = audit.audit_candidates(**inputs(tmp_path, gold, five, qlora))
    assert result["totals"]["tp_gain"] == 2
    assert result["per_relation"]["hasArea"]["added_individually_unmatched"] == 1
    union = [{**f, "ObjectEntities": list(dict.fromkeys(f["ObjectEntities"] + q["ObjectEntities"]))}
             for f, q in zip(five, qlora)]
    for name, rows in (("five_shot", five), ("qlora", qlora), ("union", union)):
        scores = official.evaluate_per_sr_pair(rows, gold, official.RELATION_TYPE, 0.05)
        macro, micro = official.macro_average_per_relation(scores), official.micro_average_per_relation(scores)
        assert result["metrics"][name] == {key: {**macro[key], **micro[key]} for key in macro}
    assert result["queries"][2]["scores"]["union"]["f1"] == 1.0


def test_numeric_normalization_collision_keeps_first_string(tmp_path):
    # Official punctuation dedup precedes numeric parsing; never fix this upstream edge case.
    result = audit.audit_candidates(**inputs(tmp_path,
        [row([["100"]], relation="hasArea")], [row(["-100"], relation="hasArea")],
        [row(["100"], relation="hasArea")]))
    assert result["totals"]["normalized_additional_candidates"] == 0
    assert result["totals"]["tp_gain"] == 0
    assert result["queries"][0]["qlora_raw_additions"][0]["individually_matchable"]


def test_full_precision_aggregation_uses_official_sorted_order(tmp_path):
    gold = [row([[str(i)] for i in range(denominator)], subject)
            for subject, denominator in (("Z", 7), ("B", 3), ("A", 9), ("C", 11))]
    predictions = [{**g, "ObjectEntities": ["0", "wrong"]} for g in gold]
    result = audit.audit_candidates(**inputs(tmp_path, gold, predictions, predictions))
    scores = official.evaluate_per_sr_pair(predictions, gold, official.RELATION_TYPE, 0.05)
    assert result["metrics"]["union"]["*** All Relations ***"]["macro-f1"] == (
        official.macro_average_per_relation(scores)["*** All Relations ***"]["macro-f1"])


@pytest.mark.parametrize("which", ["gold_path", "five_shot_path", "qlora_path"])
def test_duplicate_identities_rejected(tmp_path, which):
    kwargs = inputs(tmp_path)
    content = kwargs[which].read_text(encoding="utf-8")
    kwargs[which].write_text(content + content, encoding="utf-8")
    with pytest.raises(audit.AuditError, match="duplicate"):
        audit.audit_candidates(**kwargs)


@pytest.mark.parametrize("replacement", [[], [row([], "Other")], [row([1])]])
def test_coverage_and_schema_rejected(tmp_path, replacement):
    kwargs = inputs(tmp_path)
    write(kwargs["qlora_path"], replacement)
    with pytest.raises((audit.AuditError, DatasetError)):
        audit.audit_candidates(**kwargs)


def test_order_rejected(tmp_path):
    kwargs = inputs(tmp_path, [row([], "A"), row([], "B")],
                    [row([], "B"), row([], "A")], [row([], "A"), row([], "B")])
    with pytest.raises(audit.AuditError, match="order"):
        audit.audit_candidates(**kwargs)


def test_raw_failures_metadata_and_partial_log(tmp_path):
    kwargs = inputs(tmp_path, [row([], "A"), row([["100"]], "B", "hasArea")],
                    [row([], "A"), row(["100"], "B", "hasArea")],
                    [row([], "A"), row([], "B", "hasArea")])
    failure = {"input_line": 1, "SubjectEntity": "A", "Relation": RELATION,
               "response": "24522\u27e9", "parsing_status": "parse_failed", "error": "malformed"}
    raw = write(tmp_path / "raw.jsonl", [failure])
    failures = write(tmp_path / "failures.jsonl", [failure])
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"status": "complete", "output_sha256": hashlib.sha256(
        kwargs["five_shot_path"].read_bytes()).hexdigest(), "dataset": {"sha256": hashlib.sha256(
        kwargs["gold_path"].read_bytes()).hexdigest()}}), encoding="utf-8")
    kwargs["sidecars"] = {"five_shot": {"raw": raw, "failures": failures, "metadata": metadata}}
    result = audit.audit_candidates(**kwargs)
    diagnostics = result["artifact_diagnostics"]["five_shot"]
    assert diagnostics["raw"]["complete"] is False
    assert diagnostics["raw"]["counts"]["parse_failed"] == 1
    assert diagnostics["raw"]["per_relation"][RELATION]["parse_failed"] == 1
    assert diagnostics["metadata"]["prediction_hash_verified"]
    metadata.write_text('{"output_sha256": "bad"}', encoding="utf-8")
    with pytest.raises(audit.AuditError, match="SHA256"):
        audit.audit_candidates(**kwargs)


def test_successful_raw_and_incomplete_array_diagnostic(tmp_path):
    kwargs = inputs(tmp_path, [row([["100"]], relation="hasArea")],
                    [row(["100"], relation="hasArea")], [row([], relation="hasArea")])
    common = {"input_line": 1, "SubjectEntity": "A", "Relation": "hasArea"}
    raw = write(tmp_path / "raw.jsonl", [{**common, "response": "[100]", "parsing_status": "valid", "error": None}])
    failed = write(tmp_path / "failed.jsonl", [{**common, "response": '["one",', "parsing_status": "parse_failed", "error": "bad"}])
    kwargs["sidecars"] = {"five_shot": {"raw": raw}, "qlora": {"raw": failed}}
    result = audit.audit_candidates(**kwargs)
    assert result["artifact_diagnostics"]["five_shot"]["raw"]["counts"]["numeric_normalized"] == 1
    assert result["artifact_diagnostics"]["qlora"]["raw"]["counts"]["failed_array_without_closing_bracket"] == 1
    write(raw, [{**common, "response": "[101]", "parsing_status": "valid"}])
    with pytest.raises(audit.AuditError, match="differs from predictions"):
        audit.audit_candidates(**kwargs)


def test_nulls_relation_magnitude_and_exclusions():
    gold = [row([[value]], str(i), relation) for i, (value, relation) in enumerate([
        ("100", "hasArea"), ("200", "hasArea"), ("10000", "hasArea"),
        ("0", "hasArea"), ("-1", "hasArea"), ("NaN", "hasArea"), ("100", "hasCapacity")])]
    five = [{**g, "ObjectEntities": []} for g in gold]
    union = [{**g, "ObjectEntities": [g["ObjectEntities"][0][0]]} for g in gold]
    result = audit.numeric_null_controls(official, gold, five, union, iterations=50, seed=23)
    assert result == audit.numeric_null_controls(official, gold, five, union, iterations=50, seed=23)
    area = result["hasArea"]
    assert area["eligible_queries"] == 3 and area["excluded_queries"] == 3
    assert area["within_relation"]["movable_queries"] == 3
    assert area["within_relation_magnitude_decade"]["bucket_sizes"] == {"2": 2, "4": 1}
    assert area["within_relation_magnitude_decade"]["movable_queries"] == 2
    assert result["hasCapacity"]["within_relation"]["fixed_identity_fraction"] == 1.0
    assert area["within_relation"]["observed"]["tp_gain"] == 3
    assert area["within_relation"]["null"]["tp_gain"]["mean"] < 3


def test_reference_checks_and_archived_union(tmp_path):
    kwargs = inputs(tmp_path)
    kwargs["union_path"] = write(tmp_path / "union.jsonl", [row(["FR", "fr", "France", "Spain"])])
    result = audit.audit_candidates(**kwargs, expected={"queries": 475, "tp_gain": 77})
    assert result["archived_union_comparison"] == {"exact_objects_and_order_match": True, "official_scores_match": True}
    assert result["reference_checks"]["queries"]["matches"] is False


def test_complete_raw_metadata_counts_are_verified(tmp_path):
    kwargs = inputs(tmp_path, five=[row([])])
    raw = write(tmp_path / "raw.jsonl", [{"input_line": 1, "SubjectEntity": "A", "Relation": RELATION,
                                        "response": "[]", "parsing_status": "valid", "error": None}])
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"valid_records": 0}', encoding="utf-8")
    with pytest.raises(audit.AuditError, match="valid_records"):
        audit.audit_candidates(**kwargs, sidecars={"five_shot": {"raw": raw, "metadata": metadata}})


def test_cli_no_overwrite_and_no_output_on_validation_failure(tmp_path, capsys):
    kwargs = inputs(tmp_path)
    output = tmp_path / "audit.json"
    args = ["--gold", str(kwargs["gold_path"]), "--five-shot", str(kwargs["five_shot_path"]),
            "--qlora", str(kwargs["qlora_path"]), "--output", str(output), "--null-iterations", "3"]
    assert audit.main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert "not candidate precision" in capsys.readouterr().out
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        audit.main(args)
    assert error.value.code == 2 and output.read_bytes() == before
    output.unlink()
    write(kwargs["qlora_path"], [])
    with pytest.raises(SystemExit):
        audit.main(args)
    assert not output.exists()


def test_public_test_and_invalid_null_settings_rejected(tmp_path):
    kwargs = inputs(tmp_path)
    with pytest.raises(audit.AuditError, match="iterations"):
        audit.audit_candidates(**{**kwargs, "iterations": 0})
    kwargs["gold_path"] = write(tmp_path / "test.jsonl", [row([])])
    with pytest.raises(audit.AuditError, match="validation"):
        audit.audit_candidates(**kwargs)
