"""Extract structured prostate MRI fields from reports with a local vLLM model."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

FIELDS = [
    "前列腺左右径", "前列腺前后径", "上下径", "前列腺体积", "前列腺增生",
    "增生部位", "凸向膀胱", "外周带萎缩", "双侧外周带呈", "前列腺出血",
    "双侧精囊腺", "淋巴结肿大", "可疑骨转移", "盆腔积液", "病灶数量",
    "病灶位置", "形状", "边缘", "最大径", "ADC值", "T2W评分",
    "DWI/ADC评分", "DCE", "突破包膜", "侵犯精囊腺", "尿道外括约肌侵犯",
    "直肠侵犯", "膀胱侵犯", "PI-RADS评分",
]


def make_prompt(text: str) -> str:
    schema = json.dumps({field: None for field in FIELDS}, ensure_ascii=False, indent=2)
    return (
        "你是一位资深影像科专家。请从以下MRI报告中提取字段；未提及的字段填null。"
        "只输出一个严格合法的JSON对象，不要添加解释。\n\n"
        f"输出结构：\n{schema}\n\n报告：\n{text}"
    )


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conclusion-column", default="报告结论")
    parser.add_argument("--findings-column", default="报告表现")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    frame = pd.read_excel(args.input) if args.input.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.input)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True)
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=args.max_tokens)
    rows: list[dict] = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        prompts = [make_prompt(f"报告结论：{row.get(args.conclusion_column, '')}\n报告表现：{row.get(args.findings_column, '')}") for _, row in batch.iterrows()]
        outputs = llm.generate(prompts, sampling)
        for (_, row), output in zip(batch.iterrows(), outputs):
            result = row.to_dict()
            try:
                result.update(parse_json_object(output.outputs[0].text))
                result["处理状态"] = "已处理"
            except ValueError as exc:
                result["处理状态"] = f"解析失败: {exc}"
            rows.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    cli()

