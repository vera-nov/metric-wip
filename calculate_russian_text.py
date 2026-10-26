from __future__ import annotations

import json
import hashlib
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from model_loading import generation_model_class, require_complete_generation

from utils import (
    CANDIDATE_COL,
    REFERENCE_COL,
    ROW_ID_COL,
    cleanup_cuda,
    configure_worker_environment,
    get_input_device,
    pick_dtype,
    require_column,
)

PROMPT_WORD_LIMIT = 300
MAX_MODEL_LENGTH = 8192
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
# Compatible with the GREEN journal produced by commit 14ebc87. Older valid
# answers should survive this reliability-only change to the source file.
LEGACY_GREEN_SOURCE_SHA256 = "a65477e7e261095a5bb7c58a560891f1bd9c94d9dcdc15447dc2f16ae636ddda"


class GreenGenerationLimitError(ValueError):
    """Keep all decoded outputs, including incomplete ones, for diagnosis."""

    def __init__(self, outputs: List[str], complete: List[bool]):
        super().__init__("GREEN generation hit max_new_tokens before EOS")
        self.outputs = outputs
        self.complete = complete


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def safe_apply_qwen_chat_template(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


@dataclass
class ParsedGreen:
    score: float
    sig_total: int
    insig_total: int
    matched_findings: int
    errors: Dict[str, Any]


class ManualGREENQwen:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = get_input_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {
            "torch_dtype": pick_dtype(),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device != "cpu" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self.model = generation_model_class(model_name).from_pretrained(model_name, **model_kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def make_prompt(self, reference_text: str, candidate_text: str) -> str:
        return f"""Цель: Оценить точность проверяемого радиологического отчета по сравнению с эталонным радиологическим отчетом, написанным экспертами-радиологами.

Обзор процесса:

Ты получишь:

1. Критерии оценки.
2. Эталонный радиологический отчет.
3. Проверяемый радиологический отчет.
4. Требуемый формат ответа.

1. Критерии оценки:

Для проверяемого отчета определи:

- количество клинически значимых ошибок
- количество клинически незначимых ошибок

Возможные категории ошибок:
(а) Ложное указание находки в проверяемом отчете
(б) Пропуск находки, присутствующей в эталонном отчете
(в) Неверное определение анатомической локализации или положения находки
(г) Неверная оценка степени выраженности находки
(д) Упоминание сравнения, которого нет в эталонном отчете
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием

Сосредоточься на клинических находках, а не на стиле изложения. Оценивай только находки, которые присутствуют в отчетах.

2. Эталонный отчет:
{reference_text}
3. Проверяемый отчет:
{candidate_text}

4. Представление оценки:

Строго следуй этому формату, даже если ошибок не найдено:

[Объяснение]:

<краткое объяснение>

[Клинически значимые ошибки]:

(а) Ложное указание находки в проверяемом отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(б) Пропуск находки, присутствующей в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(в) Неверное определение анатомической локализации или положения находки: <число>. <ошибка 1>; <ошибка 2>; ...
(г) Неверная оценка степени выраженности находки: <число>. <ошибка 1>; <ошибка 2>; ...
(д) Упоминание сравнения, которого нет в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием: <число>. <ошибка 1>; <ошибка 2>; ...

[Клинически незначимые ошибки]:

(а) Ложное указание находки в проверяемом отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(б) Пропуск находки, присутствующей в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(в) Неверное определение анатомической локализации или положения находки: <число>. <ошибка 1>; <ошибка 2>; ...
(г) Неверная оценка степени выраженности находки: <число>. <ошибка 1>; <ошибка 2>; ...
(д) Упоминание сравнения, которого нет в эталонном отчете: <число>. <ошибка 1>; <ошибка 2>; ...
(е) Пропуск сравнения, описывающего изменение по сравнению с предыдущим исследованием: <число>. <ошибка 1>; <ошибка 2>; ...

[Совпадающие находки]:

<число>. <находка 1>; <находка 2>; ...

В каждом из двух разделов ошибок обязательно перечисли все шесть категорий (а)–(е).
Для отсутствующих ошибок укажи число 0 после двоеточия. Не заменяй категории
общей фразой об отсутствии ошибок. В разделе совпадающих находок сначала
укажи целое неотрицательное число. Сохрани все заголовки в квадратных скобках."""

    def clean_response(self, text: str) -> str:
        return strip_thinking(text).strip()

    def _tokenize_batch(self, prompts: Sequence[str]):
        messages = [[{"role": "user", "content": p}] for p in prompts]
        rendered = [safe_apply_qwen_chat_template(self.tokenizer, m) for m in messages]
        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        if batch["input_ids"].shape[1] > MAX_MODEL_LENGTH:
            raise ValueError("GREEN input exceeds context budget; refusing to truncate report")
        return batch

    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        import torch

        batch = self._tokenize_batch(prompts)
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        input_len = batch["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = [self.clean_response(self.tokenizer.decode(o[input_len:], skip_special_tokens=True))
                   for o in output]
        complete = []
        for o in output:
            try:
                require_complete_generation(o[input_len:], max_new_tokens, self.tokenizer.eos_token_id)
                complete.append(True)
            except ValueError:
                complete.append(False)
        if not all(complete):
            raise GreenGenerationLimitError(decoded, complete)
        return decoded

    def _extract_section(self, text: str, title: str) -> str:
        pattern = rf"\[{re.escape(title)}\]\s*:\s*(.*?)(?=\n\s*\[[^\]]+\]\s*:|\Z)"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
        if len(matches) > 1:
            raise ValueError(f"Duplicate GREEN section: {title}")
        return matches[0].group(1).strip() if matches else ""

    def _parse_count_and_desc(self, line: str) -> tuple[int, str]:
        match = re.search(r":\s*(\d+)", line)
        count = int(match.group(1)) if match else 0
        desc = ""
        if match:
            desc = line[match.end() :].strip(" .;\t")
        return count, desc

    def parse(self, text: str) -> ParsedGreen:
        text = strip_thinking(text)
        text = re.sub(r"\*\*|__|`", "", text)
        text = text.replace("：", ":").replace("\u00a0", " ")
        # Accept Markdown and a colon inside brackets without changing counts.
        for title in ("Объяснение", "Клинически значимые ошибки",
                      "Клинически незначимые ошибки", "Совпадающие находки"):
            text = re.sub(
                rf"^\s*(?:\#{{1,6}}\s*)?(?:\[{title}\s*:?\s*\]|{title})\s*:?\s*$",
                f"[{title}]:", text, flags=re.MULTILINE | re.IGNORECASE,
            )

        sig_section = self._extract_section(
            text, "Клинически значимые ошибки"
        )
        insig_section = self._extract_section(
            text, "Клинически незначимые ошибки"
        )
        matched_section = self._extract_section(
            text, "Совпадающие находки"
        )

        if not sig_section or not insig_section or not matched_section:
            raise ValueError("Incomplete GREEN answer: missing sections")

        errors = {"significant": {}, "insignificant": {}}
        alphabet = {
            **{c: c for c in "абвгде"},
            **dict(zip("abcdef", "абвгде")),
        }
        marker = r"^\s*(?:[-*•]\s*)?(?:\(([абвгдеabcdef])\)|([абвгдеabcdef])[.)])\s*"
        expected = {"(" + c + ")" for c in "абвгде"}

        for label, section in (
            ("significant", sig_section),
            ("insignificant", insig_section),
        ):
            adjective = "значимых" if label == "significant" else "незначимых"
            zero_patterns = (
                rf"(?:Клинически\s+)?{adjective}\s+ошибок\s+нет",
                rf"Нет\s+(?:клинически\s+)?{adjective}\s+ошибок",
                rf"(?:Клинически\s+)?{adjective}\s+ошибок\s+не\s+(?:выявлено|обнаружено)",
                rf"(?:Клинически\s+)?{adjective[:-1]}е\s+ошибки\s+отсутствуют",
                r"Ошибок\s+нет|Нет\s+ошибок|Ошибки\s+отсутствуют|Не\s+(?:выявлено|обнаружено)|Нет|Отсутствуют|0",
            )
            if re.fullmatch(r"\s*(?:" + "|".join(zero_patterns) + r")[.!]?\s*",
                            section, re.IGNORECASE):
                errors[label] = {key: {"count": 0, "description": ""} for key in expected}
                continue
            matches = list(
                re.finditer(marker, section, re.MULTILINE | re.IGNORECASE)
            )

            for i, match in enumerate(matches):
                key = "(" + alphabet[(match.group(1) or match.group(2)).lower()] + ")"
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(section)
                )
                block = section[match.end():end]

                # Counts must occur immediately after the marker or category's colon.
                # Never search explanations for arbitrary digits or invent a zero.
                count_match = re.match(r"(?:[^:\n]*:\s*)?(\d+)\b(?![.,]\d)", block)
                zero_match = re.fullmatch(
                    r"(?:[^:\n]*:\s*)?(?:нет(?:\s+ошибок)?|ошибок\s+нет|отсутствуют|"
                    r"не\s+(?:выявлено|обнаружено))[.!]?\s*", block, re.IGNORECASE,
                )
                if (count_match is None and zero_match is None) or key in errors[label]:
                    raise ValueError(
                        f"Missing count or duplicate category: {label} {key}"
                    )

                errors[label][key] = {
                    "count": int(count_match.group(1)) if count_match else 0,
                    "description": block[count_match.end():].strip(
                        " .;\t\n"
                    ) if count_match else "",
                }

            if set(errors[label]) != expected:
                raise ValueError(
                    f"GREEN error counts missing or malformed: {label}"
                )

        sig_total = sum(
            item["count"] for item in errors["significant"].values()
        )
        insig_total = sum(
            item["count"] for item in errors["insignificant"].values()
        )

        match = re.match(r"\s*(\d+)\b(?![.,]\d)", matched_section)
        if match is None:
            raise ValueError("GREEN matched-findings count is missing")

        matched = int(match.group(1))
        score = matched / (matched + sig_total) if matched else 0.0

        return ParsedGreen(
            score=score,
            sig_total=sig_total,
            insig_total=insig_total,
            matched_findings=matched,
            errors=errors,
        )

    def score_pairs(
        self,
        references: Sequence[str],
        candidates: Sequence[str],
        batch_size: int = 1,
        max_new_tokens: int = 512,
        row_ids: Optional[Sequence[str]] = None,
        response_log: Optional[Path] = None,
        cache_context: Optional[Dict[str, Any]] = None,
        legacy_cache_contexts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        if len(references) != len(candidates) or batch_size < 1:
            raise ValueError("GREEN needs equal pair counts and batch_size >= 1")
        ids = list(row_ids) if row_ids is not None else [str(i) for i in range(len(references))]
        if len(ids) != len(references) or len(set(ids)) != len(ids):
            raise ValueError("GREEN row IDs must be unique and match pair counts")
        prompts = [self.make_prompt(str(r), str(c)) for r, c in zip(references, candidates)]
        def make_keys(context, token_budget=None):
            return [hashlib.sha256(json.dumps(
                {"id": str(row_id), "prompt": prompt,
                 **({"max_new_tokens": token_budget} if token_budget is not None else {}),
                 "context": context}, sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest() for row_id, prompt in zip(ids, prompts)]

        # An answer that already ended with EOS is independent of its upper
        # token budget. Version the scoring contract, not incidental code edits.
        keys = make_keys(cache_context)
        legacy_keys = [make_keys(context, budget)
                       for context in (legacy_cache_contexts or [])
                       for budget in {max_new_tokens, 512, 1024, 2048, 4096, 8192}]
        results: Dict[int, Dict[str, Any]] = {}

        def parse_row(output):
            parsed = self.parse(output)
            return {
                "ru_GREEN_Qwen": parsed.score,
                "ru_GREEN_Qwen_sig_errors": parsed.sig_total,
                "ru_GREEN_Qwen_insig_errors": parsed.insig_total,
                "ru_GREEN_Qwen_matched_findings": parsed.matched_findings,
                "ru_GREEN_Qwen_errors_json": json.dumps(parsed.errors, ensure_ascii=False),
                "ru_GREEN_Qwen_analysis": output,
            }

        # Replay valid raw answers with the current parser. The journal is separate
        # from the worker's temporary CSV, which pipeline.py deletes on failure.
        if response_log is not None and response_log.exists():
            indices = {key: i for i, key in enumerate(keys)}
            for aliases in legacy_keys:
                indices.update({key: i for i, key in enumerate(aliases)})
            with response_log.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        warnings.warn("Ignoring damaged GREEN journal line")
                        continue
                    if (not isinstance(entry, dict) or entry.get("key") not in indices
                            or entry.get("status", "complete") != "complete"):
                        continue
                    try:
                        results[indices[entry["key"]]] = parse_row(entry["output"])
                    except (ValueError, KeyError, TypeError):
                        continue
        pending = [i for i in range(len(prompts)) if i not in results]
        print(f"GREEN: reused {len(results)} answers; pending {len(pending)}", flush=True)
        failed = {}

        def record(i, output, attempt, *, complete=True):
            if response_log is not None:
                response_log.parent.mkdir(parents=True, exist_ok=True)
                with response_log.open("a", encoding="utf-8") as stream:
                    # Leading newline also separates a partial line after interruption.
                    stream.write("\n" + json.dumps({"key": keys[i], "report_id": str(ids[i]),
                                 "attempt": attempt, "status": "complete" if complete else "truncated",
                                 "output": output}, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            if not complete:
                failed[i] = f"Generation hit max_new_tokens={max_new_tokens} before EOS"
                return
            try:
                results[i] = parse_row(output)
                failed.pop(i, None)
            except ValueError as exc:
                failed[i] = str(exc)

        def generate_and_record(indices, batch_prompts, attempt):
            try:
                outputs = self.generate_batch(batch_prompts, max_new_tokens=max_new_tokens)
                complete = [True] * len(outputs)
            except GreenGenerationLimitError as exc:
                outputs, complete = exc.outputs, exc.complete
            if len(outputs) != len(indices) or len(complete) != len(indices):
                raise ValueError("GREEN generated the wrong number of answers")
            for i, output, finished in zip(indices, outputs, complete):
                record(i, output, attempt, complete=finished)

        started = time.monotonic()
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            generate_and_record(indices, [prompts[i] for i in indices], 0)
            for i in indices:
                if i not in failed:
                    continue
                retry_prompt = prompts[i] + (
                    "\nОтветь максимально кратко, без повторений и длинных объяснений. "
                    "В каждом разделе ошибок нужны ровно шесть строк "
                    "(а)–(е) с целым числом после двоеточия, в том числе при числе 0. "
                    "В разделе [Совпадающие находки] сначала укажи целое число. "
                    "Не пропускай разделы и категории."
                )
                generate_and_record([i], [retry_prompt], 1)
            done = start + len(indices)
            elapsed = time.monotonic() - started
            eta = elapsed / done * (len(pending) - done) if done else 0
            print(f"GREEN: {len(results)}/{len(prompts)} valid, "
                  f"{done}/{len(pending)} attempted this run, "
                  f"{len(failed)} failed, elapsed {duration(elapsed)}, "
                  f"ETA {duration(eta)}", flush=True)
        if failed:
            details = {str(ids[i]): error for i, error in failed.items()}
            raise ValueError(f"GREEN answers still malformed: {details}. "
                             f"Raw answers: {response_log}. Valid answers are retained; rerun with --resume.")
        return pd.DataFrame([results[i] for i in range(len(prompts))])


def calculate_green_for_csv(
    input_csv: str,
    output_csv: str,
    model_name: str = DEFAULT_MODEL,
    candidate_col: str = CANDIDATE_COL,
    reference_col: str = REFERENCE_COL,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: Optional[str] = None,
    gpu: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    configure_worker_environment(gpu, device)
    df = pd.read_csv(input_csv, dtype={ROW_ID_COL: str}, keep_default_na=False)
    for col in (ROW_ID_COL, candidate_col, reference_col):
        require_column(df, col)
    scorer = ManualGREENQwen(model_name=model_name, device=device)
    scores = scorer.score_pairs(
        references=df[reference_col].fillna("").astype(str).tolist(),
        candidates=df[candidate_col].fillna("").astype(str).tolist(),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        row_ids=df[ROW_ID_COL].astype(str).tolist(),
        response_log=Path(input_csv).resolve().parent / "green.responses.jsonl",
        cache_context={
            "model": model_name, "device": device, "gpu": gpu, "batch_size": batch_size,
            "source": "green-qwen-1", "model_loading": "strict-complete-1",
        },
        legacy_cache_contexts=[{
            "model": model_name, "device": device, "gpu": gpu, "batch_size": batch_size,
            "source": LEGACY_GREEN_SOURCE_SHA256,
            "model_loading": hashlib.sha256(Path(__file__).with_name("model_loading.py").read_bytes()).hexdigest(),
        }],
    )
    out = pd.concat([df[[ROW_ID_COL]].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    out.to_csv(output_csv, index=False)
    cleanup_cuda(scorer)
    return out
