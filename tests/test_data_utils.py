import json

from profound.data.extract_reports import parse_json_object
from profound.data.prepare_single_case import (
    build_inference_record,
    infer_series_type,
    parse_series_map,
)


def test_parse_fenced_json():
    assert parse_json_object('```json\n{"PI-RADS评分": 4}\n```') == {"PI-RADS评分": 4}


def test_sample_infer_is_valid_json():
    with open("examples/sample_infer.jsonl", encoding="utf-8") as handle:
        record = json.loads(next(handle))
    placeholders = record["messages"][1]["content"].count("<video>")
    assert placeholders == len(record["videos"])


def test_single_case_modality_inference():
    assert infer_series_type("subject_x_T2W") == "T2W"
    assert infer_series_type("subject_x_DWI") == "DWI"
    assert infer_series_type("subject_x_ADC") == "ADC"
    assert infer_series_type("unknown_sequence") is None


def test_series_map_parser():
    assert parse_series_map(["folder_a=T2W", "folder_b=adc"]) == {
        "folder_a": "T2W",
        "folder_b": "ADC",
    }


def test_single_case_inference_record_has_matching_videos():
    series = [
        {"series_type": "T2W", "video_path": "/data/t2w.mp4"},
        {"series_type": "DWI", "video_path": "/data/dwi.mp4"},
        {"series_type": "ADC", "video_path": "/data/adc.mp4"},
    ]
    record = build_inference_record("case-012", series)
    placeholders = record["messages"][-2]["content"].count("<video>")
    assert placeholders == len(record["videos"]) == 3
    assert record["messages"][-1] == {"role": "assistant", "content": ""}
