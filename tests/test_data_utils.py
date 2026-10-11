import json

from profound.data.extract_reports import parse_json_object


def test_parse_fenced_json():
    assert parse_json_object('```json\n{"PI-RADS评分": 4}\n```') == {"PI-RADS评分": 4}


def test_sample_sft_is_valid_json():
    with open("examples/sample_sft.jsonl", encoding="utf-8") as handle:
        record = json.loads(next(handle))
    placeholders = record["messages"][1]["content"].count("<video>")
    assert placeholders == len(record["videos"])
