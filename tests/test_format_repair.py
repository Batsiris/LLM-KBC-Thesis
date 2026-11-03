"""Synthetic post-processing checks; no models, gold labels, or metrics used."""

import hashlib
import json
from pathlib import Path

import pytest

from llm_kbc import format_repair as fr
from llm_kbc.dataset_inspector import DatasetError
from llm_kbc.prediction_io import load_predictions
from llm_kbc.zero_shot import InferenceError, parse_response


def artifacts(tmp_path, responses):
    raw, baseline = [], []
    for index, response in enumerate(responses, start=1):
        identity = {"SubjectEntity": f"Subject {index}", "Relation": "hasArea" if index % 2 else "awardWonBy"}
        try:
            objects = parse_response(response)
        except InferenceError as error:
            objects, status, message = [], "parse_failed", str(error)
        else:
            status, message = "valid", None
        raw.append({**identity, "input_line": index, "response": response, "parsing_status": status, "error": message})
        baseline.append({**identity, "ObjectEntities": objects, "metadata": {"original": True}})
    raw_path, baseline_path = tmp_path / "original.raw.jsonl", tmp_path / "baseline.jsonl"
    write_rows(raw_path, raw)
    # Deliberately noncanonical whitespace/CRLF: valid rows must stay byte-exact.
    baseline_path.write_bytes(b"".join(
        ("  " + json.dumps(row, ensure_ascii=False) + " \r\n").encode("utf-8") for row in baseline
    ))
    return raw_path, baseline_path


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("token", ["104.4", "418", "-0", "-0.0", "1.2300", "4.2E-7", "1e400", "9007199254740993"])
def test_standalone_json_number_preserves_spelling(token):
    assert fr.repair_response(f" \n{token}\t") == ([token], "standalone_number", None)


@pytest.mark.parametrize("response, expected", [
    ("[France, Germany]", ["France", "Germany"]),
    ("[O'Brien, Baden-Württemberg]", ["O'Brien", "Baden-Württemberg"]),
    ("[St. Louis, O\u2019Brien, Åland Islands]", ["St. Louis", "O\u2019Brien", "Åland Islands"]),
    # Accepted by the fixed grammar despite documented semantic ambiguity.
    ("[Paris, Texas]", ["Paris", "Texas"]),
    ("[Acme, Incorporated]", ["Acme", "Incorporated"]),
])
def test_conservative_unquoted_entity_lists(response, expected):
    assert fr.repair_response(response) == (expected, "unquoted_entity_list", None)


@pytest.mark.parametrize("response", [
    "Paris", "The answer is 418", "24522\u27e9", "NaN", "Infinity", "-Infinity",
    "1,234", "1,234.5", "104,4", "+418", "01", ".4", "1.", "1e", "１２３", "−12",
    "[Washington, D.C.]", "[France; Germany]",
    "[1,234]", "[1, 234]", "[1; 234]", "[France, 123]", "[France, NaN]", "[France, null]",
    "[Paris]", "[France,Germany]", "[France,  Germany]", "[France, Germany, ]",
    "[France,, Germany]", "[France; Germany, Spain]", "[France | Germany]",
    "[France/Germany, Spain]", "[France & Germany, Spain]",
    "[[France], Germany]", "[France, {Germany}]", '[France, "Germany"]', "['France', Germany]",
    "[France, Germany", "France, Germany]", "[France, Germany] explanation", "Answer: [France, Germany]",
    "[France, Germany is a country]", "[France, Germany\n]", "[France, Germany\u27e9]",
    "[France, Germany...]", "[France, Ger...many]", "[France, Ger--many]", "[France, Ger\u200bmany]",
    "```json\n418\n```", "```\n[France, Germany]\n```", "true", "null", "{}", "[]",
])
def test_ineligible_ambiguous_or_malformed_responses_are_not_repaired(response):
    objects, _, reason = fr.repair_response(response)
    assert objects is None
    assert reason


def test_complete_ablation_preserves_order_originals_and_valid_line_bytes(tmp_path):
    responses = [' [104.4, "Éva", 418] ', '  1.2300\n', '[France, Germany]', '24522\u27e9', '[]', '[1,234]']
    raw, baseline = artifacts(tmp_path, responses)
    originals = {path: path.read_bytes() for path in (raw, baseline)}
    output = tmp_path / "repaired.jsonl"
    report = fr.run_repair(raw, baseline, output)
    predictions = list(load_predictions(output))
    assert [row["SubjectEntity"] for row in predictions] == [f"Subject {i}" for i in range(1, 7)]
    assert [row["ObjectEntities"] for row in predictions] == [["104.4", "Éva", "418"], ["1.2300"], ["France", "Germany"], [], [], ["1", "234"]]
    for index in (0, 3, 4, 5):
        assert output.read_bytes().splitlines(keepends=True)[index] == originals[baseline].splitlines(keepends=True)[index]
    assert all(row["metadata"] == {"original": True} for row in predictions)
    assert all(path.read_bytes() == content for path, content in originals.items())
    audit_path = Path(str(output) + ".audit.jsonl")
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [row["input_line"] for row in audit] == [2, 3, 4]
    assert [row["raw_response"] for row in audit] == [responses[i] for i in (1, 2, 3)]
    assert [row["success"] for row in audit] == [True, True, False]
    assert [row["rule"] for row in audit] == ["standalone_number", "unquoted_entity_list", None]
    assert [row["prediction"] for row in audit] == [predictions[i] for i in (1, 2, 3)]
    assert report["total_inputs"] == 6
    assert report["valid_responses"] == report["parse_failures"] == report["repair_attempts"] == 3
    assert report["repair_successes"] == 2
    assert report["remaining_parse_failures"] == 1
    assert report["per_relation"]["hasArea"]["repair_successes"] == 1
    assert report["per_relation"]["awardWonBy"]["remaining_parse_failures"] == 1
    assert report["policy_version"] == fr.POLICY_VERSION
    assert report["exploratory"] is True
    assert report["inputs"]["baseline_predictions"]["sha256"] == hashlib.sha256(originals[baseline]).hexdigest()
    assert report["inputs"]["raw_generations"]["sha256"] == hashlib.sha256(originals[raw]).hexdigest()
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert report == json.loads(Path(str(output) + ".metadata.json").read_text())
    second = tmp_path / "second.jsonl"
    assert fr.run_repair(raw, baseline, second) == report
    assert second.read_bytes() == output.read_bytes()
    assert Path(str(second) + ".audit.jsonl").read_bytes() == audit_path.read_bytes()


def test_all_valid_responses_and_missing_final_newline_stay_byte_exact(tmp_path):
    raw, baseline = artifacts(tmp_path, ['```json\n["Paris", "Paris", ""]\n```', '[]'])
    baseline.write_bytes(baseline.read_bytes().rstrip(b"\r\n"))
    before = baseline.read_bytes()
    output = tmp_path / "output.jsonl"
    report = fr.run_repair(raw, baseline, output)
    assert output.read_bytes() == before
    assert Path(str(output) + ".audit.jsonl").read_bytes() == b""
    assert report["repair_attempts"] == report["remaining_parse_failures"] == 0


@pytest.mark.parametrize("corruption", [
    "missing_raw", "missing_baseline", "order", "duplicate", "index", "boolean_index",
    "status", "missing_error", "false_failure", "nonempty_abstention", "wrong_valid_prediction",
    "raw_not_object", "missing_identity", "nonstring_response", "invalid_schema", "invalid_error", "false_valid",
])
def test_invalid_artifacts_publish_nothing(tmp_path, corruption):
    raw, baseline = artifacts(tmp_path, ['["Paris"]', '418'])
    raw_rows = [json.loads(line) for line in raw.read_text().splitlines()]
    baseline_rows = list(load_predictions(baseline))
    if corruption == "missing_raw": raw_rows.pop()
    elif corruption == "missing_baseline": baseline_rows.pop()
    elif corruption == "order": baseline_rows.reverse()
    elif corruption == "duplicate":
        for rows in (raw_rows, baseline_rows):
            rows[1].update(SubjectEntity=rows[0]["SubjectEntity"], Relation=rows[0]["Relation"])
    elif corruption == "index": raw_rows[1]["input_line"] = 3
    elif corruption == "boolean_index": raw_rows[0]["input_line"] = True
    elif corruption == "status": raw_rows[1]["parsing_status"] = "interrupted"
    elif corruption == "missing_error": del raw_rows[1]["error"]
    elif corruption == "false_failure": raw_rows[0].update(parsing_status="parse_failed", error="false")
    elif corruption == "nonempty_abstention": baseline_rows[1]["ObjectEntities"] = ["418"]
    elif corruption == "wrong_valid_prediction": baseline_rows[0]["ObjectEntities"] = []
    elif corruption == "raw_not_object": raw_rows[1] = []
    elif corruption == "missing_identity": del raw_rows[1]["SubjectEntity"]
    elif corruption == "nonstring_response": raw_rows[1]["response"] = 418
    elif corruption == "invalid_schema": baseline_rows[1]["ObjectEntities"] = [418]
    elif corruption == "invalid_error": raw_rows[1]["error"] = None
    elif corruption == "false_valid": raw_rows[1].update(parsing_status="valid", error=None)
    write_rows(raw, raw_rows)
    write_rows(baseline, baseline_rows)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises((fr.RepairError, DatasetError)):
        fr.run_repair(raw, baseline, tmp_path / "output.jsonl")
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("bad_line", [b"{broken}\n", b"\n", b"\xff\n"])
@pytest.mark.parametrize("target", ["raw", "baseline"])
def test_malformed_input_files_fail_with_line_context(tmp_path, bad_line, target):
    raw, baseline = artifacts(tmp_path, ['418'])
    path = raw if target == "raw" else baseline
    path.write_bytes(bad_line)
    with pytest.raises(fr.RepairError, match=":1: invalid UTF-8 JSONL"):
        fr.run_repair(raw, baseline, tmp_path / "output.jsonl")
    assert not (tmp_path / "output.jsonl").exists()


def test_empty_files_are_not_complete_runs(tmp_path):
    raw, baseline = artifacts(tmp_path, [])
    with pytest.raises(fr.RepairError, match="nonzero record counts"):
        fr.run_repair(raw, baseline, tmp_path / "output.jsonl")


@pytest.mark.parametrize("suffix", ["", ".audit.jsonl", ".metadata.json"])
def test_existing_outputs_are_never_overwritten(tmp_path, suffix):
    raw, baseline = artifacts(tmp_path, ['418'])
    output = tmp_path / "output.jsonl"
    protected = Path(str(output) + suffix)
    protected.write_bytes(b"keep")
    with pytest.raises(fr.RepairError, match="Refusing to overwrite"):
        fr.run_repair(raw, baseline, output)
    assert protected.read_bytes() == b"keep"


def test_input_paths_cannot_be_used_as_outputs(tmp_path):
    raw, baseline = artifacts(tmp_path, ['418'])
    for path in (raw, baseline):
        before = path.read_bytes()
        with pytest.raises(fr.RepairError, match="Refusing to overwrite"):
            fr.run_repair(raw, baseline, path)
        assert path.read_bytes() == before


def test_publication_race_never_overwrites_existing_output(tmp_path, monkeypatch):
    raw, baseline = artifacts(tmp_path, ['418'])
    output = tmp_path / "output.jsonl"
    original_link = fr.os.link

    def race(source, target):
        if target == output:
            output.write_bytes(b"concurrent output")
        original_link(source, target)

    monkeypatch.setattr(fr.os, "link", race)
    with pytest.raises(FileExistsError):
        fr.run_repair(raw, baseline, output)
    assert output.read_bytes() == b"concurrent output"
    assert {path.name for path in tmp_path.iterdir()} == {
        raw.name, baseline.name, output.name, output.name + ".audit.jsonl", output.name + ".metadata.json",
    }


def test_publication_error_keeps_inputs_and_leaves_no_partial_prediction(tmp_path, monkeypatch):
    raw, baseline = artifacts(tmp_path, ['418'])
    originals = {path: path.read_bytes() for path in (raw, baseline)}
    output = tmp_path / "output.jsonl"
    original_link = fr.os.link

    def fail(source, target):
        if target == output:
            raise OSError("simulated publication failure")
        original_link(source, target)

    monkeypatch.setattr(fr.os, "link", fail)
    with pytest.raises(OSError, match="simulated publication failure"):
        fr.run_repair(raw, baseline, output)
    assert not output.exists()
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert {path.name for path in tmp_path.iterdir()} == {
        raw.name, baseline.name, output.name + ".audit.jsonl", output.name + ".metadata.json",
    }


def test_cli_reports_counts_and_input_errors(tmp_path, capsys):
    raw, baseline = artifacts(tmp_path, ['418'])
    args = ["--raw-generations", str(raw), "--baseline-predictions", str(baseline), "--output", str(tmp_path / "output.jsonl")]
    assert fr.main(args) == 0
    assert json.loads(capsys.readouterr().out)["repair_successes"] == 1
    with pytest.raises(SystemExit) as error:
        fr.main(args)
    assert error.value.code == 2
