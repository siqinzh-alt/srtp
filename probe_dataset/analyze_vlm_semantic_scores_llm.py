from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


SYSTEM_PROMPT = """You are a strict but semantic-focused judge for acoustic VLM evaluation.
Judge whether each model answer is semantically equivalent to the reference answer, not whether the wording or output format is exactly identical.

Scoring rules:
1. Ignore case, punctuation, JSON code fences, extra explanation, JSON key order, and Chinese/English synonym wording.
2. For single-field tasks, judge only the requested field.
3. For multi-field JSON tasks, judge each field independently. Extract understandable field values even if the JSON formatting is imperfect.
4. Numeric fields may use different unit formatting. Main frequency in Hz may have a small reading tolerance, but a wrong option or wrong order of magnitude is incorrect.
5. For fsk_order, answers like "4" and "4FSK" are equivalent, and "2" and "2FSK" are equivalent.
6. For lfm_direction, up-sweep / upward / ascending / increasing frequency are equivalent; down-sweep / downward / descending / decreasing frequency are equivalent.
7. For signal_type, CW / continuous wave / single-frequency / pure tone are equivalent; LFM / chirp / linear FM are equivalent; FSK and BPSK should be judged by semantic equivalence.
8. For channel_condition and noise_level, accept natural-language synonyms, e.g. direct can be described as clear/good/stable/direct path; real_noise_light can be described as light noise / low noise.
9. For multiple-choice tasks, count as correct if the answer clearly points to the correct option. If it gives multiple mutually exclusive options without a clear final choice, count it wrong.
10. For animal tasks: bio_species_code is an internal dataset code and is correct only when the same code is explicitly given; bio_species_or_group accepts semantically equivalent Chinese, English, or Latin species/group names; bio_call_description accepts semantically equivalent call-type descriptions.

Return only valid JSON matching the schema. Do not include extra text."""


USER_PROMPT = """Please semantically grade the following VLM answers.

For each sample, return:
- id: sample id
- field_results: one object per requested field, with field, correct, reason

The correct boolean means whether that field is semantically correct. Keep reason short.

Samples:
{items_json}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_codex_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or Path.home() / ".codex" / "config.toml"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    config: dict[str, Any] = {}
    current_section: list[str] = []
    providers: dict[str, dict[str, str]] = defaultdict(dict)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = [part.strip('"\'') for part in line.strip("[]").split(".")]
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        if not current_section:
            if key in {"model", "model_provider"}:
                config[key] = value
        elif len(current_section) == 2 and current_section[0] == "model_providers":
            providers[current_section[1]][key] = value

    selected = config.get("model_provider")
    if selected and selected in providers:
        config.update({f"provider_{key}": value for key, value in providers[selected].items()})
    return config


def api_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def load_api_key(api_key_env: str, prefer_codex_auth: bool) -> str | None:
    auth_path = Path.home() / ".codex" / "auth.json"
    if prefer_codex_auth and auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            key = auth.get("OPENAI_API_KEY")
            if isinstance(key, str) and key:
                return key
        except json.JSONDecodeError:
            pass
    return os.environ.get(api_key_env)


def compact_sample(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task": row.get("task"),
        "answer_format": row.get("answer_format"),
        "prompt": row.get("prompt"),
        "fields": fields,
        "expected_answer": row.get("expected_answer"),
        "choices": row.get("choices", []),
        "label": row.get("label", {}),
        "condition": row.get("condition", {}),
        "model_response": row.get("response", ""),
    }


def filtered_fields(row: dict[str, Any], exclude_fields: set[str]) -> list[str]:
    return [field for field in row.get("fields", []) if field not in exclude_fields]


def build_schema(batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [item["id"] for item in batch_items]
    return {
        "type": "json_schema",
        "name": "vlm_semantic_judgement_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "minItems": len(batch_items),
                    "maxItems": len(batch_items),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "enum": ids},
                            "field_results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "field": {"type": "string"},
                                        "correct": {"type": "boolean"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["field", "correct", "reason"],
                                },
                            },
                        },
                        "required": ["id", "field_results"],
                    },
                }
            },
            "required": ["results"],
        },
    }


def extract_response_text(response_json: dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]
    chunks: list[str] = []
    for output in response_json.get("output", []) or []:
        for content in output.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_responses_api(
    *,
    url: str,
    api_key: str,
    model: str,
    batch_items: list[dict[str, Any]],
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(items_json=json.dumps(batch_items, ensure_ascii=False, indent=2))},
        ],
        "text": {"format": build_schema(batch_items)},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with requests.Session() as session:
                response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            text = extract_response_text(response.json())
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"LLM judge failed after {max_retries + 1} attempts: {last_error}")


def normalize_judgement(row: dict[str, Any], judged: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    by_field = {item.get("field"): item for item in judged.get("field_results", []) if isinstance(item, dict)}
    field_results = []
    for field in fields:
        item = by_field.get(field)
        if item is None:
            field_results.append({"field": field, "correct": False, "reason": "Judge did not return this field; counted as incorrect."})
        else:
            field_results.append({"field": field, "correct": bool(item.get("correct")), "reason": str(item.get("reason", ""))})
    return {
        "id": row["id"],
        "task": row.get("task"),
        "answer_format": row.get("answer_format"),
        "fields": fields,
        "field_count": len(fields),
        "field_signature": "+".join(fields),
        "hard_correct": bool(row.get("correct")),
        "strict_correct": all(item["correct"] for item in field_results),
        "field_results": field_results,
        "expected_answer": row.get("expected_answer"),
        "response": row.get("response", ""),
    }


def metric(correct: int, total: int) -> dict[str, Any]:
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 6) if total else 0.0}


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_groups: dict[str, Counter] = defaultdict(Counter)
    field_groups: dict[str, Counter] = defaultdict(Counter)

    def add_row(group: str, row: dict[str, Any]) -> None:
        row_groups[group]["total"] += 1
        row_groups[group]["hard"] += int(row["hard_correct"])
        row_groups[group]["strict"] += int(row["strict_correct"])
        for field_result in row["field_results"]:
            row_groups[group]["field_total"] += 1
            row_groups[group]["field_correct"] += int(field_result["correct"])

    for row in scored_rows:
        groups = [
            "overall:overall",
            f"answer_format:{row.get('answer_format', 'unknown')}",
            f"field_count:{row.get('field_count', 'unknown')}",
            f"task:{row.get('task', 'unknown')}",
            f"fields:{'+'.join(row.get('fields', []))}",
        ]
        for group in groups:
            add_row(group, row)
        for field_result in row["field_results"]:
            field = field_result["field"]
            field_groups[f"field:{field}"]["total"] += 1
            field_groups[f"field:{field}"]["correct"] += int(field_result["correct"])

    summary: dict[str, Any] = {}
    for key, value in sorted(row_groups.items()):
        summary[key] = {
            "samples": value["total"],
            "hard": metric(value["hard"], value["total"]),
            "strict": metric(value["strict"], value["total"]),
            "mean": metric(value["field_correct"], value["field_total"]),
        }
    for key, value in sorted(field_groups.items()):
        summary[key] = metric(value["correct"], value["total"])
    return summary


def make_review_examples(scored_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    for row in scored_rows:
        if row["strict_correct"] and not row["hard_correct"]:
            examples.append(row)
        elif not row["strict_correct"] and any(item["correct"] for item in row["field_results"]):
            examples.append(row)
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    config = load_codex_config()
    default_model = config.get("model", "copilot-default-gpt-5.5")
    default_base_url = config.get("provider_base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"))
    parser = argparse.ArgumentParser(description="LLM semantic scoring for VLM prediction JSONL files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Detailed scored JSONL output.")
    parser.add_argument("--summary", type=Path, required=True, help="Summary JSON output.")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--no-codex-auth", action="store_true", help="Do not read ~/.codex/auth.json before the API key env var.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1, help="Parallel judge requests. Use 1 for sequential requests.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-field", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=None, help="Optional judge temperature. Omitted by default for Codex proxy models.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--examples", type=Path, default=None)
    parser.add_argument("--example-limit", type=int, default=80)
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_env, prefer_codex_auth=not args.no_codex_auth)
    if not api_key:
        raise SystemExit(f"Missing API key: ~/.codex/auth.json or environment variable {args.api_key_env}")

    rows = read_jsonl(args.predictions)
    if args.limit is not None:
        rows = rows[: args.limit]
    exclude_fields = set(args.exclude_field)
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in read_jsonl(args.output):
            existing[row["id"]] = row

    pending = []
    skipped = 0
    for row in rows:
        fields = filtered_fields(row, exclude_fields)
        if not fields:
            skipped += 1
            continue
        if row["id"] not in existing:
            pending.append((row, fields))

    print(f"judge_model={args.model}")
    print(f"judge_url={api_url(args.base_url)}")
    print(f"rows_total={len(rows)} rows_existing={len(existing)} rows_pending={len(pending)} rows_skipped={skipped} exclude_fields={sorted(exclude_fields)}")

    pending_batches = [pending[start : start + args.batch_size] for start in range(0, len(pending), args.batch_size)]

    def judge_batch(batch_pairs: list[tuple[dict[str, Any], list[str]]]) -> list[dict[str, Any]]:
        batch_items = [compact_sample(row, fields) for row, fields in batch_pairs]
        judged = call_responses_api(
            url=api_url(args.base_url),
            api_key=api_key,
            model=args.model,
            batch_items=batch_items,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        judged_by_id = {item.get("id"): item for item in judged.get("results", []) if isinstance(item, dict)}
        return [normalize_judgement(row, judged_by_id.get(row["id"], {"field_results": []}), fields) for row, fields in batch_pairs]

    def flush_outputs() -> None:
        ordered = [existing[row["id"]] for row in rows if row["id"] in existing]
        write_jsonl(args.output, ordered)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summarize(ordered), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.examples:
            args.examples.write_text(json.dumps(make_review_examples(ordered, args.example_limit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.workers <= 1:
        for batch_pairs in tqdm(pending_batches, desc="llm semantic judge"):
            for scored in judge_batch(batch_pairs):
                existing[scored["id"]] = scored
            flush_outputs()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(judge_batch, batch_pairs) for batch_pairs in pending_batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc="llm semantic judge"):
                for scored in future.result():
                    existing[scored["id"]] = scored
                flush_outputs()

    ordered = [existing[row["id"]] for row in rows if row["id"] in existing]
    write_jsonl(args.output, ordered)
    summary = summarize(ordered)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.examples:
        args.examples.write_text(json.dumps(make_review_examples(ordered, args.example_limit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary.get("overall:overall", {}), ensure_ascii=False))
    print(f"Scored: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
