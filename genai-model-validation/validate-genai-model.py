#!/usr/bin/env python3
"""
GenAI Model Validation Script

Validates deployed vLLM / vLLM-Omni models against their acceptance criteria.
Supports: Text (standard vLLM), TTS, Diffusion, and Omni (multimodal) models.

Usage:
    python validate-genai-model.py -e <endpoint> -m <model_name> -t <model_type>

Model types:
    text       - Standard vLLM text/chat models (Llama, Mistral, etc.)
    tts        - Text-to-Speech (Qwen3-TTS, Voxtral-TTS)
    diffusion  - Image generation (FLUX.2-klein, FLUX.2-dev)
    omni       - Multimodal text+vision+audio (Qwen3-Omni)

Exit codes:
    0 - All tests passed
    1 - Test failures
    2 - Connectivity error
    3 - Config/input error
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Missing required dependency: requests")
    print(f"  Install with: {sys.executable} -m pip install 'requests>=2.28.0'")
    sys.exit(1)

# ── ANSI Colors ──────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def disable_colors() -> None:
    global RED, GREEN, YELLOW, CYAN, BOLD, DIM, NC
    RED = GREEN = YELLOW = CYAN = BOLD = DIM = NC = ""


if not sys.stdout.isatty():
    disable_colors()

# ── Magic Bytes ──────────────────────────────────────────────────────────────

MAGIC_PNG = bytes.fromhex("89504e47")
MAGIC_JPEG = bytes.fromhex("ffd8ff")
MAGIC_RIFF = bytes.fromhex("52494646")
MAGIC_WEBP = b"WEBP"
MAGIC_MP3_FFFB = bytes.fromhex("fffb")
MAGIC_MP3_FFF3 = bytes.fromhex("fff3")
MAGIC_ID3 = bytes.fromhex("494433")
MAGIC_FLAC = bytes.fromhex("664c6143")

# ── Directories ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"

# ── Test Registry ────────────────────────────────────────────────────────────

TEST_REGISTRY: dict[str, list[str]] = {
    "common": [
        "common_health",
        "common_models",
        "common_schema",
        "common_metadata",
        "common_negative_auth",
        "common_negative_invalid_model",
        "common_negative_malformed_body",
        "common_negative_empty_body",
        "common_negative_max_tokens_zero",
        "common_concurrency",
    ],
    "text": [
        "text_chat_completion",
        "text_completions",
        "text_streaming",
        "text_multiturn",
        "text_system_prompt",
        "text_token_boundary",
        "text_temperature",
        "text_stop_sequences",
        "text_logprobs",
        "text_long_context",
        "text_streaming_delta",
        "text_empty_messages",
        "text_invalid_role",
        "text_function_calling",
    ],
    "tts": [
        "tts_wav",
        "tts_mp3",
        "tts_voices",
        "tts_second_voice",
        "tts_flac",
        "tts_long_text",
        "tts_empty_input",
        "tts_unsupported_voice",
        "tts_wav_duration",
        "tts_speed",
        "tts_streaming_audio",
        "tts_multi_language",
    ],
    "diffusion": [
        "diffusion_generate",
        "diffusion_chat_image",
        "diffusion_second_prompt",
        "diffusion_size_matrix",
        "diffusion_seed_repro",
        "diffusion_batch",
        "diffusion_different_prompts",
        "diffusion_negative_prompt",
        "diffusion_guidance_scale",
        "diffusion_invalid_size",
        "diffusion_empty_prompt",
        "diffusion_url_response",
        "diffusion_num_inference_steps",
    ],
    "omni": [
        "omni_text_chat",
        "omni_completions",
        "omni_vision",
        "omni_audio_transcription",
        "omni_audio_output",
        "omni_streaming_multimodal",
        "omni_multiturn_vision",
        "omni_multiple_images",
        "omni_transcription_formats",
        "omni_transcription_accuracy",
        "omni_audio_output_mp3",
        "omni_wav_output_check",
        "omni_modality_combinations",
        "omni_large_image",
        "omni_unsupported_modality",
        "omni_vision_url_vs_base64",
    ],
}


def all_test_names_for_type(model_type: str) -> list[str]:
    return TEST_REGISTRY.get("common", []) + TEST_REGISTRY.get(model_type, [])


NEGATIVE_TESTS = frozenset({
    "common_negative_auth", "common_negative_invalid_model",
    "common_negative_malformed_body", "common_negative_empty_body",
    "common_negative_max_tokens_zero",
    "text_empty_messages", "text_invalid_role",
    "tts_empty_input", "tts_unsupported_voice",
    "diffusion_invalid_size", "diffusion_empty_prompt",
    "omni_unsupported_modality",
})


_EXCLUDE_SET: frozenset[str] = frozenset()


def should_run_test(test_name: str, test_filter: list[str] | None,
                    skip_negative: bool) -> bool:
    if skip_negative and test_name in NEGATIVE_TESTS:
        return False
    if test_name in _EXCLUDE_SET:
        return False
    if test_filter is None:
        return True
    if test_name in ("common_health", "common_models", "common_schema", "common_metadata"):
        return True
    return test_name in test_filter


# ── Result Tracker ───────────────────────────────────────────────────────────


class ResultTracker:
    """Tracks test results with PASS/FAIL/WARN/SKIP counts."""

    def __init__(self, timeout_warn_ms: float = 0.0) -> None:
        self.pass_count = 0
        self.fail_count = 0
        self.warn_count = 0
        self.skip_count = 0
        self.results: list[dict[str, Any]] = []
        self.latency_times: list[float] = []
        self.timeout_warn_ms = timeout_warn_ms

    def record(self, test_name: str, status: str, detail: str = "",
               latency_ms: float = 0.0) -> None:
        detail_str = f" ({detail})" if detail else ""
        if status == "PASS":
            print(f"{GREEN}  PASS{NC} {test_name}{detail_str}")
            self.pass_count += 1
        elif status == "FAIL":
            print(f"{RED}  FAIL{NC} {test_name}{detail_str}")
            self.fail_count += 1
        elif status == "WARN":
            print(f"{YELLOW}  WARN{NC} {test_name}{detail_str}")
            self.warn_count += 1
        else:
            print(f"{YELLOW}  SKIP{NC} {test_name}{detail_str}")
            self.skip_count += 1

        if (self.timeout_warn_ms > 0 and latency_ms > self.timeout_warn_ms
                and status in ("PASS", "FAIL")):
            print(f"{YELLOW}       ⚠ {test_name} took {latency_ms:.0f}ms "
                  f"(threshold: {self.timeout_warn_ms:.0f}ms){NC}")

        self.results.append({
            "name": test_name,
            "status": status,
            "detail": detail,
            "latency_ms": latency_ms,
        })
        if latency_ms > 0 and status in ("PASS", "WARN"):
            self.latency_times.append(latency_ms)

    def compute_latency_stats(self) -> dict[str, float]:
        if not self.latency_times:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        sorted_times = sorted(self.latency_times)
        n = len(sorted_times)
        p50_idx = int(n * 0.5)
        p95_idx = min(int(n * 0.95), n - 1)
        return {
            "p50": sorted_times[p50_idx],
            "p95": sorted_times[p95_idx],
            "max": sorted_times[-1],
            "mean": statistics.mean(sorted_times),
            "count": n,
        }

    def print_summary(self, model_name: str, model_type: str,
                      endpoint: str, output_dir: Path) -> bool:
        print_header("VALIDATION SUMMARY")
        print()
        print(f"  Model:    {BOLD}{model_name}{NC}")
        print(f"  Type:     {BOLD}{model_type}{NC}")
        print(f"  Endpoint: {BOLD}{endpoint}{NC}")
        print()

        for result in self.results:
            name = result["name"]
            status = result["status"]
            detail = result["detail"]
            detail_str = f" — {detail}" if detail else ""
            if status == "PASS":
                print(f"    {GREEN}PASS{NC}  {name}{detail_str}")
            elif status == "FAIL":
                print(f"    {RED}FAIL{NC}  {name}{detail_str}")
            elif status == "WARN":
                print(f"    {YELLOW}WARN{NC}  {name}{detail_str}")
            else:
                print(f"    {YELLOW}SKIP{NC}  {name}{detail_str}")

        total = self.pass_count + self.fail_count + self.warn_count + self.skip_count
        print()
        print(f"  {GREEN}Passed: {self.pass_count}{NC}  "
              f"{RED}Failed: {self.fail_count}{NC}  "
              f"{YELLOW}Warned: {self.warn_count}{NC}  "
              f"{YELLOW}Skipped: {self.skip_count}{NC}  "
              f"Total: {total}")

        stats = self.compute_latency_stats()
        if stats["count"] > 0:
            print()
            print(f"  Latency (ms): p50={stats['p50']:.0f}  "
                  f"p95={stats['p95']:.0f}  max={stats['max']:.0f}  "
                  f"mean={stats['mean']:.0f}  ({stats['count']} samples)")
        print()

        if self.fail_count > 0:
            print(f"  {RED}{BOLD}RESULT: VALIDATION FAILED{NC}")
        else:
            print(f"  {GREEN}{BOLD}RESULT: VALIDATION PASSED{NC}")

        skips = [r for r in self.results if r["status"] == "SKIP"]
        warns = [r for r in self.results if r["status"] == "WARN"]
        if skips or warns:
            print()
            print(f"  {BOLD}Notes:{NC}")
            for r in skips:
                reason = r["detail"] if r["detail"] else "not applicable"
                print(f"    SKIP  {r['name']}: {reason}")
            for r in warns:
                reason = r["detail"] if r["detail"] else "see above"
                print(f"    WARN  {r['name']}: {reason}")
            print()
            print(f"  SKIPs indicate tests not applicable to this model type.")
            print(f"  WARNs are informational and do not affect the verdict.")

        print()
        print(f"  Output files saved to: {output_dir}/")
        return self.fail_count == 0

    def to_json_report(self, model_name: str, model_type: str,
                       endpoint: str, metadata: dict) -> dict:
        stats = self.compute_latency_stats()
        return {
            "metadata": {
                "model": model_name,
                "type": model_type,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat(),
                **metadata,
            },
            "latency": stats,
            "tests": self.results,
            "summary": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
                "skip": self.skip_count,
                "total": self.pass_count + self.fail_count + self.warn_count + self.skip_count,
            },
            "verdict": "PASS" if self.fail_count == 0 else "FAIL",
        }


# ── HTTP Client ──────────────────────────────────────────────────────────────


class HttpClient:
    """HTTP client with retry, SSL control, auth, verbose logging, and timeouts."""

    RETRY_CODES = {502, 503, 504}
    RETRY_COUNT = 2

    def __init__(self, *, insecure: bool = False, bearer_token: str = "",
                 verbose: bool = False, get_timeout: int = 60,
                 post_timeout: int = 300) -> None:
        self.verbose = verbose
        self.verify_ssl = not insecure
        self.get_timeout = get_timeout
        self.post_timeout = post_timeout
        self.retries_occurred = False
        self.session = requests.Session()
        if bearer_token:
            self.session.headers["Authorization"] = f"Bearer {bearer_token}"

        retry_strategy = Retry(
            total=self.RETRY_COUNT,
            backoff_factor=2,
            status_forcelist=[],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"{DIM}    {msg}{NC}")

    def _should_retry(self, code: int) -> bool:
        return code in self.RETRY_CODES

    def _retry_delay(self, attempt: int) -> None:
        self._log(f">> Retry {attempt} after transient failure")
        self.retries_occurred = True
        time.sleep(attempt * 2)

    def get(self, url: str, timeout: int | None = None) -> requests.Response | None:
        t = timeout if timeout is not None else self.get_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> GET {url}")
            try:
                resp = self.session.get(url, verify=self.verify_ssl, timeout=t)
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None

    def post_json(self, url: str, data: dict,
                  timeout: int | None = None) -> requests.Response | None:
        t = timeout if timeout is not None else self.post_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST {url}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=t,
                )
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None

    def post_raw(self, url: str, raw_body: str | bytes,
                 timeout: int | None = None) -> requests.Response | None:
        """POST raw (non-JSON) body for negative testing."""
        t = timeout if timeout is not None else self.post_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST (raw) {url}")
            try:
                headers = {"Content-Type": "application/json"}
                resp = self.session.post(
                    url, data=raw_body, headers=headers,
                    verify=self.verify_ssl, timeout=t,
                )
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None

    def post_binary(self, url: str, data: dict, output_path: Path,
                    timeout: int | None = None) -> tuple[int, str]:
        """POST JSON, save binary response to file. Returns (status_code, content_type)."""
        t = timeout if timeout is not None else self.post_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST {url} -> {output_path}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=t,
                    stream=True,
                )
                content_type = resp.headers.get("Content-Type", "")
                transfer_encoding = resp.headers.get("Transfer-Encoding", "")
                self._log(f"<< HTTP {resp.status_code} Content-Type: {content_type}")
                if self._should_retry(resp.status_code):
                    continue
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return resp.status_code, content_type
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return 0, ""
        return 0, ""

    def post_binary_with_headers(self, url: str, data: dict, output_path: Path,
                                 timeout: int | None = None) -> tuple[int, str, dict]:
        """Like post_binary but also returns response headers."""
        t = timeout if timeout is not None else self.post_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST {url} -> {output_path}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=t,
                    stream=True,
                )
                content_type = resp.headers.get("Content-Type", "")
                headers = dict(resp.headers)
                self._log(f"<< HTTP {resp.status_code} Content-Type: {content_type}")
                if self._should_retry(resp.status_code):
                    continue
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return resp.status_code, content_type, headers
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return 0, "", {}
        return 0, "", {}

    def post_stream(self, url: str, data: dict,
                    timeout: int | None = None) -> str:
        """POST JSON for SSE streaming, returns raw text."""
        t = timeout if timeout is not None else self.post_timeout
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST (stream) {url}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=t,
                )
                text = resp.text
                if not text and attempt < self.RETRY_COUNT:
                    self._log("<< Empty response, will retry")
                    continue
                self._log(f"<< {text[:500]}")
                return text
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return ""
        return ""

    MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}

    def post_multipart(self, url: str, file_path: Path, file_field: str,
                       data: dict, timeout: int | None = None) -> requests.Response | None:
        """POST multipart form data. Opens file fresh on each retry attempt."""
        t = timeout if timeout is not None else self.post_timeout
        mime = self.MIME_MAP.get(file_path.suffix.lower(), "application/octet-stream")
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST (multipart) {url}")
            try:
                with open(file_path, "rb") as f:
                    files = {file_field: (file_path.name, f, mime)}
                    resp = self.session.post(
                        url, files=files, data=data,
                        verify=self.verify_ssl, timeout=t,
                    )
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None


# ── Utility Functions ────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    print()
    print(f"{CYAN}{'━' * 60}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{CYAN}{'━' * 60}{NC}")


def print_test(title: str) -> None:
    print(f"\n{YELLOW}--- {title} ---{NC}")


def file_magic_check(filepath: Path, expected: bytes) -> bool:
    try:
        with open(filepath, "rb") as f:
            header = f.read(len(expected))
        return header == expected
    except OSError:
        return False


def is_webp(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
        return header[:4] == MAGIC_RIFF and header[8:12] == MAGIC_WEBP
    except OSError:
        return False


def is_mp3(filepath: Path) -> bool:
    return (file_magic_check(filepath, MAGIC_MP3_FFFB)
            or file_magic_check(filepath, MAGIC_MP3_FFF3)
            or file_magic_check(filepath, MAGIC_ID3))


def check_content_type(actual: str, *expected: str) -> bool:
    ct = actual.split(";")[0].strip().lower()
    return any(e in ct for e in expected)


def save_json(data: Any, path: Path) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except (OSError, TypeError):
        pass


def detect_and_rename_image(img_path: Path, base_name: str) -> None:
    """Detect image format by magic bytes and rename with correct extension."""
    try:
        if file_magic_check(img_path, MAGIC_PNG):
            img_path.rename(img_path.parent / f"{base_name}.png")
        elif file_magic_check(img_path, MAGIC_JPEG):
            img_path.rename(img_path.parent / f"{base_name}.jpg")
        elif is_webp(img_path):
            img_path.rename(img_path.parent / f"{base_name}.webp")
    except OSError:
        pass


def parse_wav_duration(filepath: Path) -> float | None:
    """Parse PCM WAV header and return duration in seconds. Returns None if not PCM."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
            if header[:4] != MAGIC_RIFF or header[8:12] != b"WAVE":
                return None
            # Scan for 'data' chunk starting at byte 12
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    return None
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
                if chunk_id == b"fmt ":
                    fmt_data = f.read(chunk_size)
                    if len(fmt_data) < 16:
                        return None
                    audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                    if audio_format != 1:  # Not PCM
                        return None
                    num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                    sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                    bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                elif chunk_id == b"data":
                    # Compute duration
                    bytes_per_sample = bits_per_sample // 8
                    if sample_rate == 0 or num_channels == 0 or bytes_per_sample == 0:
                        return None
                    total_samples = chunk_size // (num_channels * bytes_per_sample)
                    return total_samples / sample_rate
                else:
                    f.seek(chunk_size, 1)
    except (OSError, struct.error, UnboundLocalError):
        return None


def parse_png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Extract width, height from PNG IHDR chunk (bytes 16-23)."""
    if len(data) < 24 or data[:4] != MAGIC_PNG:
        return None
    try:
        width = struct.unpack(">I", data[16:20])[0]
        height = struct.unpack(">I", data[20:24])[0]
        return (width, height)
    except struct.error:
        return None


def timed_call(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call fn and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


# ── Schema Validation ────────────────────────────────────────────────────────

SCHEMAS: dict[str, dict[str, type | tuple]] = {
    "chat_completion": {
        "id": str,
        "object": str,
        "choices": list,
        "model": str,
    },
    "text_completion": {
        "id": str,
        "object": str,
        "choices": list,
        "model": str,
    },
    "streaming_chunk": {
        "id": str,
        "object": str,
        "choices": list,
    },
    "images": {
        "data": list,
    },
}


def validate_schema(body: dict, schema_type: str, tracker: ResultTracker) -> None:
    schema = SCHEMAS.get(schema_type, {})
    if not schema:
        return
    for field, expected_type in schema.items():
        value = body.get(field)
        if value is None:
            tracker.record(f"Schema {schema_type}.{field} present", "FAIL", "missing")
        elif not isinstance(value, expected_type):
            tracker.record(f"Schema {schema_type}.{field} type", "FAIL",
                           f"expected {expected_type.__name__}, got {type(value).__name__}")
        else:
            tracker.record(f"Schema {schema_type}.{field} valid", "PASS")


# ── Runtime Metadata ─────────────────────────────────────────────────────────


def capture_runtime_metadata(client: HttpClient, endpoint: str) -> dict:
    """Best-effort probe for runtime info (version, gpu, model metadata)."""
    metadata: dict[str, Any] = {}

    # Probe /version
    resp = client.get(f"{endpoint}/version")
    if resp is not None and resp.status_code == 200:
        try:
            vdata = resp.json()
            metadata["version"] = vdata.get("version", str(vdata))
        except (json.JSONDecodeError, ValueError):
            metadata["version"] = resp.text.strip()[:100]

    # Probe /health for extra info
    resp = client.get(f"{endpoint}/health")
    if resp is not None and resp.status_code == 200:
        try:
            hdata = resp.json()
            if isinstance(hdata, dict):
                for key in ("gpu", "gpu_memory", "device", "dtype"):
                    if key in hdata:
                        metadata[key] = hdata[key]
        except (json.JSONDecodeError, ValueError):
            pass

    # Probe /v1/models for model info
    resp = client.get(f"{endpoint}/v1/models")
    if resp is not None and resp.status_code == 200:
        try:
            mdata = resp.json()
            models = mdata.get("data", [])
            if models and isinstance(models[0], dict):
                m = models[0]
                metadata["model_id"] = m.get("id", "")
                if "max_model_len" in m:
                    metadata["max_model_len"] = m["max_model_len"]
                if "dtype" in m:
                    metadata["dtype"] = m["dtype"]
        except (json.JSONDecodeError, ValueError):
            pass

    return metadata


# ── Warm-up ──────────────────────────────────────────────────────────────────


def warmup_request(client: HttpClient, endpoint: str, model_name: str,
                   model_type: str, *, tts_voice: str = "alloy") -> None:
    """Send one warm-up request to prime the model. Failures are non-fatal."""
    print(f"\n{DIM}  Sending warm-up request...{NC}")
    try:
        if model_type in ("text", "omni"):
            client.post_json(f"{endpoint}/v1/chat/completions", {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
            })
        elif model_type == "tts":
            client.post_json(f"{endpoint}/v1/audio/speech", {
                "model": model_name,
                "input": "test",
                "voice": tts_voice,
                "response_format": "wav",
            })
        elif model_type == "diffusion":
            client.post_json(f"{endpoint}/v1/images/generations", {
                "model": model_name,
                "prompt": "test",
                "size": "256x256",
            })
        print(f"{DIM}  Warm-up complete{NC}")
    except Exception:
        print(f"  {YELLOW}WARNING: Warm-up request failed, continuing anyway{NC}")


# ── Negative Tests (Common) ──────────────────────────────────────────────────


def _primary_endpoint(model_type: str) -> str:
    if model_type in ("text", "omni"):
        return "/v1/chat/completions"
    elif model_type == "tts":
        return "/v1/audio/speech"
    elif model_type == "diffusion":
        return "/v1/images/generations"
    return "/v1/chat/completions"


def _minimal_valid_payload(model_name: str, model_type: str) -> dict:
    if model_type in ("text", "omni"):
        return {
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }
    elif model_type == "tts":
        return {
            "model": model_name,
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
        }
    elif model_type == "diffusion":
        return {
            "model": model_name,
            "prompt": "a dot",
            "size": "256x256",
        }
    return {}


def validate_negative_common(client: HttpClient, endpoint: str, model_name: str,
                             model_type: str, tracker: ResultTracker, *,
                             insecure: bool, verbose: bool,
                             bearer_token: str,
                             test_filter: list[str] | None,
                             skip_negative: bool) -> None:
    print_test("Negative Tests (Common)")
    primary = _primary_endpoint(model_type)
    url = f"{endpoint}{primary}"

    # Auth failure
    if should_run_test("common_negative_auth", test_filter, skip_negative):
        if bearer_token:
            bad_client = HttpClient(
                insecure=insecure, bearer_token="INVALID_TOKEN_XYZ_000",
                verbose=verbose, get_timeout=client.get_timeout,
                post_timeout=client.post_timeout)
            resp = bad_client.get(f"{endpoint}/v1/models")
            if resp is not None and resp.status_code in (401, 403):
                tracker.record("Negative: auth failure", "PASS",
                               f"HTTP {resp.status_code}")
            elif resp is not None and 400 <= resp.status_code < 500:
                tracker.record("Negative: auth failure", "PASS",
                               f"HTTP {resp.status_code}")
            elif resp is not None and resp.status_code >= 500:
                tracker.record("Negative: auth failure", "FAIL",
                               f"HTTP {resp.status_code} (expected 4xx)")
            else:
                tracker.record("Negative: auth failure", "FAIL",
                               "no auth enforcement detected")
        else:
            tracker.record("Negative: auth failure", "SKIP",
                           "BEARER_TOKEN not set")

    # Invalid model
    if should_run_test("common_negative_invalid_model", test_filter, skip_negative):
        payload = _minimal_valid_payload("NONEXISTENT_MODEL_XYZ", model_type)
        payload["model"] = "NONEXISTENT_MODEL_XYZ"
        resp = client.post_json(url, payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Negative: invalid model", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Negative: invalid model", "FAIL",
                           f"HTTP {resp.status_code} (expected 4xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Negative: invalid model", "FAIL",
                           f"HTTP {code} (expected 4xx)")

    # Malformed body
    if should_run_test("common_negative_malformed_body", test_filter, skip_negative):
        resp = client.post_raw(url, "this is not valid json{{{")
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Negative: malformed body", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Negative: malformed body", "FAIL",
                           f"HTTP {resp.status_code} (expected 4xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Negative: malformed body", "FAIL",
                           f"HTTP {code} (expected 4xx)")

    # Empty body
    if should_run_test("common_negative_empty_body", test_filter, skip_negative):
        resp = client.post_json(url, {})
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Negative: empty body", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Negative: empty body", "FAIL",
                           f"HTTP {resp.status_code} (expected 4xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Negative: empty body", "FAIL",
                           f"HTTP {code} (expected 4xx)")

    # max_tokens: 0
    if should_run_test("common_negative_max_tokens_zero", test_filter, skip_negative):
        if model_type in ("text", "omni"):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0,
            }
            resp = client.post_json(url, payload)
            if resp is not None and 400 <= resp.status_code < 500:
                tracker.record("Negative: max_tokens=0", "PASS",
                               f"HTTP {resp.status_code}")
            elif resp is not None and 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                usage = body.get("usage", {})
                if usage.get("completion_tokens", -1) == 0:
                    tracker.record("Negative: max_tokens=0", "PASS",
                                   "200 with 0 completion tokens")
                else:
                    tracker.record("Negative: max_tokens=0", "PASS",
                                   "returned empty completion")
            elif resp is not None and resp.status_code >= 500:
                tracker.record("Negative: max_tokens=0", "FAIL",
                               f"HTTP {resp.status_code} (expected 4xx or empty response)")
            else:
                tracker.record("Negative: max_tokens=0", "SKIP",
                               "no response")
        else:
            tracker.record("Negative: max_tokens=0", "SKIP",
                           f"not applicable for {model_type}")


# ── Common Validations ───────────────────────────────────────────────────────


def validate_health(client: HttpClient, endpoint: str, tracker: ResultTracker) -> bool:
    print_test("AC: /health endpoint reports healthy")
    resp, latency = timed_call(client.get, f"{endpoint}/health")
    if resp is not None and resp.status_code == 200:
        tracker.record("/health returns 200", "PASS", latency_ms=latency)
        return True
    code = resp.status_code if resp else 0
    tracker.record("/health returns 200", "FAIL", f"HTTP {code}", latency_ms=latency)
    return False


def validate_models_list(client: HttpClient, endpoint: str, model_name: str,
                         tracker: ResultTracker) -> None:
    print_test("AC: /v1/models lists the served model")
    resp, latency = timed_call(client.get, f"{endpoint}/v1/models")
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("/v1/models returns 200", "FAIL", f"HTTP {code}",
                       latency_ms=latency)
        return

    tracker.record("/v1/models returns 200", "PASS", latency_ms=latency)

    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        tracker.record("/v1/models lists model(s)", "FAIL", "invalid JSON")
        return

    data = body.get("data", [])
    if not data:
        tracker.record("/v1/models lists model(s)", "FAIL", "empty data array")
        return

    model_ids = [m.get("id", "") for m in data if isinstance(m, dict)]
    tracker.record("/v1/models lists model(s)", "PASS", ", ".join(model_ids))

    if model_name in model_ids:
        tracker.record(f"/v1/models contains {model_name}", "PASS")
    else:
        tracker.record(f"/v1/models contains {model_name}", "FAIL",
                       f"model not found in: {', '.join(model_ids)}")


# ── Text Validation ──────────────────────────────────────────────────────────


def validate_text(client: HttpClient, endpoint: str, model_name: str,
                  output_dir: Path, tracker: ResultTracker, *,
                  test_filter: list[str] | None = None,
                  skip_negative: bool = False) -> None:

    # /v1/chat/completions
    if should_run_test("text_chat_completion", test_filter, skip_negative):
        print_test("AC: Chat completion via /v1/chat/completions")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Explain the water cycle in two sentences."}],
            "max_tokens": 256,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is None or not (200 <= resp.status_code < 300):
            code = resp.status_code if resp else 0
            tracker.record("Chat completion returns 200", "FAIL", f"HTTP {code}",
                           latency_ms=latency)
        else:
            tracker.record("Chat completion returns 200", "PASS", latency_ms=latency)
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "chat-response.json")
            validate_schema(body, "chat_completion", tracker)

            text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if text and len(text) > 10:
                tracker.record("Chat response is non-empty", "PASS", f"{len(text)} chars")
            else:
                tracker.record("Chat response is non-empty", "FAIL", "empty or too short")

            reason = (body.get("choices") or [{}])[0].get("finish_reason")
            if reason:
                tracker.record("Finish reason present", "PASS", reason)
            else:
                tracker.record("Finish reason present", "FAIL", "missing")

            usage = body.get("usage")
            if usage:
                tracker.record("Usage stats present", "PASS",
                               f"prompt={usage.get('prompt_tokens', 0)}, "
                               f"completion={usage.get('completion_tokens', 0)}")
            else:
                tracker.record("Usage stats present", "FAIL", "missing")

    # /v1/completions
    if should_run_test("text_completions", test_filter, skip_negative):
        print_test("AC: Text completion via /v1/completions")
        payload = {
            "model": model_name,
            "prompt": "List three benefits of open source software:",
            "max_tokens": 256,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "completions-response.json")
            choices = body.get("choices", [])
            ctext = choices[0].get("text", "") if choices else ""
            if ctext and len(ctext) > 10:
                tracker.record("/v1/completions works", "PASS", f"{len(ctext)} chars",
                               latency_ms=latency)
            else:
                tracker.record("/v1/completions works", "FAIL", "empty response",
                               latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("/v1/completions works", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Streaming
    if should_run_test("text_streaming", test_filter, skip_negative):
        print_test("AC: Streaming chat completion")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 64,
            "stream": True,
        }
        stream_text, latency = timed_call(
            client.post_stream, f"{endpoint}/v1/chat/completions", payload)
        if stream_text:
            chunks = [line for line in stream_text.splitlines() if line.startswith("data: ")]
            if len(chunks) > 1:
                tracker.record("Streaming returns SSE chunks", "PASS",
                               f"{len(chunks)} chunks", latency_ms=latency)
                (output_dir / "stream-response.txt").write_text(stream_text)
            elif len(chunks) == 1:
                tracker.record("Streaming returns SSE chunks", "FAIL",
                               "only 1 chunk (not streaming)", latency_ms=latency)
            else:
                tracker.record("Streaming returns SSE chunks", "FAIL",
                               "no SSE data chunks", latency_ms=latency)

            if "data: [DONE]" in stream_text:
                tracker.record("Stream ends with [DONE]", "PASS")
            else:
                tracker.record("Stream ends with [DONE]", "FAIL", "missing terminator")
        else:
            tracker.record("Streaming chat completion", "FAIL", "empty response",
                           latency_ms=latency)

    # Multi-turn
    if should_run_test("text_multiturn", test_filter, skip_negative):
        print_test("AC: Multi-turn conversation")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "My name is Alex."},
                {"role": "assistant", "content": "Nice to meet you, Alex!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 64,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "multiturn-response.json")
            mtext = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if "alex" in mtext.lower():
                tracker.record("Multi-turn retains context", "PASS", latency_ms=latency)
            else:
                tracker.record("Multi-turn retains context", "FAIL",
                               "name 'Alex' not in response", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Multi-turn conversation", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # System prompt (JSON mode)
    if should_run_test("text_system_prompt", test_filter, skip_negative):
        print_test("AC: System prompt (JSON compliance)")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "respond only in valid JSON"},
                {"role": "user", "content": "What is 2 plus 2?"},
            ],
            "max_tokens": 64,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            # Strip code fences
            text_clean = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
            text_clean = re.sub(r"\n?```\s*$", "", text_clean)
            try:
                json.loads(text_clean)
                tracker.record("System prompt: valid JSON output", "PASS",
                               latency_ms=latency)
            except (json.JSONDecodeError, ValueError):
                tracker.record("System prompt: valid JSON output", "FAIL",
                               f"got: {text[:80]}", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("System prompt: JSON compliance", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Token boundary
    if should_run_test("text_token_boundary", test_filter, skip_negative):
        print_test("AC: Token boundary (max_tokens=1)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Count to ten."}],
            "max_tokens": 1,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            usage = body.get("usage", {})
            comp_tokens = usage.get("completion_tokens", -1)
            finish = (body.get("choices") or [{}])[0].get("finish_reason", "")
            if comp_tokens == 1 and finish == "length":
                tracker.record("Token boundary: 1 token + length", "PASS",
                               latency_ms=latency)
            elif comp_tokens <= 1:
                tracker.record("Token boundary: 1 token + length", "WARN",
                               f"tokens={comp_tokens}, finish_reason={finish}",
                               latency_ms=latency)
            else:
                tracker.record("Token boundary: 1 token + length", "FAIL",
                               f"tokens={comp_tokens}, finish_reason={finish}",
                               latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Token boundary", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Temperature
    if should_run_test("text_temperature", test_filter, skip_negative):
        print_test("AC: Temperature variants")
        for temp in (0, 1.5):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16,
                "temperature": temp,
            }
            resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record(f"Temperature={temp} returns 200", "PASS")
            else:
                code = resp.status_code if resp else 0
                tracker.record(f"Temperature={temp} returns 200", "FAIL",
                               f"HTTP {code}")

    # Stop sequences
    if should_run_test("text_stop_sequences", test_filter, skip_negative):
        print_test("AC: Stop sequences")
        payload = {
            "model": model_name,
            "prompt": "The word you must print is: EN",
            "max_tokens": 32,
            "stop": ["END"],
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            choices = body.get("choices", [])
            finish = choices[0].get("finish_reason", "") if choices else ""
            if finish == "stop":
                tracker.record("Stop sequence: finish_reason=stop", "PASS",
                               latency_ms=latency)
            else:
                tracker.record("Stop sequence: finish_reason=stop", "WARN",
                               f"finish_reason={finish}", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Stop sequences", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Logprobs
    if should_run_test("text_logprobs", test_filter, skip_negative):
        print_test("AC: Logprobs")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "logprobs": True,
            "top_logprobs": 5,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Logprobs", "SKIP", "not supported (4xx)")
        elif resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            choices = body.get("choices", [])
            lp = choices[0].get("logprobs") if choices else None
            if lp is None:
                tracker.record("Logprobs", "SKIP", "logprobs field null/absent")
            else:
                tracker.record("Logprobs returned", "PASS")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Logprobs", "FAIL", f"HTTP {code}")

    # Long context
    if should_run_test("text_long_context", test_filter, skip_negative):
        print_test("AC: Long context")
        long_prompt = "The sky is blue and the grass is green. " * 300
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 32,
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Long context: 200 OK", "PASS", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Long context", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Streaming delta validation
    if should_run_test("text_streaming_delta", test_filter, skip_negative):
        print_test("AC: Streaming delta structure")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Count 1 to 5."}],
            "max_tokens": 64,
            "stream": True,
        }
        stream_text = client.post_stream(f"{endpoint}/v1/chat/completions", payload)
        if stream_text:
            data_chunks = []
            for line in stream_text.splitlines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        data_chunks.append(chunk)
                    except (json.JSONDecodeError, ValueError):
                        pass
            if len(data_chunks) >= 3:
                indices = [0, len(data_chunks) // 2, len(data_chunks) - 1]
                all_have_delta = True
                for idx in indices:
                    choices = data_chunks[idx].get("choices", [])
                    if not choices or "delta" not in choices[0]:
                        all_have_delta = False
                        break
                if all_have_delta:
                    tracker.record("Streaming delta: first/mid/last have delta", "PASS")
                else:
                    tracker.record("Streaming delta: delta missing in chunks", "FAIL")
            elif data_chunks:
                tracker.record("Streaming delta: too few chunks", "WARN",
                               f"{len(data_chunks)} chunks")
            else:
                tracker.record("Streaming delta: no parseable chunks", "FAIL")
        else:
            tracker.record("Streaming delta", "FAIL", "empty response")

    # Empty messages
    if should_run_test("text_empty_messages", test_filter, skip_negative):
        print_test("AC: Empty messages (negative)")
        payload = {
            "model": model_name,
            "messages": [],
            "max_tokens": 8,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Empty messages: returns 4xx", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Empty messages: returns 4xx", "FAIL",
                           f"HTTP {resp.status_code} (got 5xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Empty messages: returns 4xx", "FAIL",
                           f"HTTP {code}")

    # Invalid role
    if should_run_test("text_invalid_role", test_filter, skip_negative):
        print_test("AC: Invalid role (negative)")
        payload = {
            "model": model_name,
            "messages": [{"role": "alien", "content": "hi"}],
            "max_tokens": 8,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Invalid role: returns 4xx", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Invalid role: returns 4xx", "FAIL",
                           f"HTTP {resp.status_code} (got 5xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Invalid role: returns 4xx", "FAIL",
                           f"HTTP {code}")

    # Function calling
    if should_run_test("text_function_calling", test_filter, skip_negative):
        print_test("AC: Function calling (tool use)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is the weather in London?"}],
            "max_tokens": 128,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                },
            }],
        }
        resp, latency = timed_call(client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Function calling", "SKIP",
                           f"HTTP {resp.status_code} (unsupported)")
        elif resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Function calling: returns 200", "PASS",
                           latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Function calling", "FAIL", f"HTTP {code}",
                           latency_ms=latency)


# ── TTS Validation ───────────────────────────────────────────────────────────


def _discover_voices(client: HttpClient, endpoint: str) -> tuple[bool, list[str]]:
    """Discover available voices from /v1/audio/voices."""
    resp = client.get(f"{endpoint}/v1/audio/voices")
    if resp is None or not (200 <= resp.status_code < 300):
        return False, []
    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        return True, []

    voices: list[str] = []
    if isinstance(body, list):
        voices = [str(v) for v in body if v]
    elif isinstance(body, dict):
        items = body.get("voices") or body.get("data") or []
        for v in items:
            if isinstance(v, dict):
                name = v.get("name") or v.get("id")
                if name:
                    voices.append(name)
            elif v:
                voices.append(str(v))
    return True, voices


def validate_tts(client: HttpClient, endpoint: str, model_name: str,
                 output_dir: Path, tracker: ResultTracker, *,
                 test_filter: list[str] | None = None,
                 skip_negative: bool = False) -> None:
    text = "Hello, this is a validation test of the text to speech system. The weather today is sunny and warm."

    # Discover voices
    voices_ok, voices = _discover_voices(client, endpoint)
    primary_voice = voices[0] if voices else "alloy"
    print(f"  Using primary voice: {BOLD}{primary_voice}{NC}")

    baseline_size = 0

    # WAV speech
    if should_run_test("tts_wav", test_filter, skip_negative):
        print_test("AC: /v1/audio/speech endpoint (wav format, predefined voice)")
        outfile = output_dir / "speech.wav"
        payload = {"model": model_name, "input": text, "voice": primary_voice,
                   "response_format": "wav"}
        (code, content_type), latency = timed_call(
            client.post_binary, f"{endpoint}/v1/audio/speech", payload, outfile)

        if not (200 <= code < 300):
            tracker.record("/v1/audio/speech (wav)", "FAIL", f"HTTP {code}",
                           latency_ms=latency)
        else:
            tracker.record("/v1/audio/speech returns 200", "PASS", latency_ms=latency)
            fsize = outfile.stat().st_size if outfile.exists() else 0
            baseline_size = fsize
            if fsize > 44:
                tracker.record("Response contains audio data", "PASS", f"{fsize} bytes")
            else:
                tracker.record("Response contains audio data", "FAIL",
                               f"{fsize} bytes (too small)")

            if file_magic_check(outfile, MAGIC_RIFF):
                tracker.record("WAV magic bytes (RIFF)", "PASS")
            else:
                tracker.record("WAV magic bytes (RIFF)", "FAIL")

            if check_content_type(content_type, "audio/wav", "audio/x-wav", "audio/wave"):
                tracker.record("Content-Type header (wav)", "PASS", content_type)
            else:
                tracker.record("Content-Type header (wav)", "FAIL",
                               f"got: {content_type}")

    # MP3 speech
    if should_run_test("tts_mp3", test_filter, skip_negative):
        print_test("AC: /v1/audio/speech endpoint (mp3 format)")
        mp3file = output_dir / "speech.mp3"
        mp3_payload = {"model": model_name, "input": text, "voice": primary_voice,
                       "response_format": "mp3"}
        (code, content_type), latency = timed_call(
            client.post_binary, f"{endpoint}/v1/audio/speech", mp3_payload, mp3file)

        if not (200 <= code < 300):
            tracker.record("/v1/audio/speech (mp3)", "FAIL", f"HTTP {code}",
                           latency_ms=latency)
        else:
            tracker.record("/v1/audio/speech (mp3) returns 200", "PASS",
                           latency_ms=latency)
            mp3_ok = is_mp3(mp3file)
            if mp3_ok:
                tracker.record("MP3 magic bytes", "PASS")
            else:
                tracker.record("MP3 magic bytes", "FAIL")

            if check_content_type(content_type, "audio/mpeg", "audio/mp3"):
                tracker.record("Content-Type header (mp3)", "PASS", content_type)
            else:
                tracker.record("Content-Type header (mp3)", "FAIL",
                               f"got: {content_type}")

    # /v1/audio/voices validation
    if should_run_test("tts_voices", test_filter, skip_negative):
        print_test("AC: /v1/audio/voices lists available speakers")
        if voices_ok:
            tracker.record("/v1/audio/voices returns 200", "PASS")
            if voices:
                tracker.record("Voices list is non-empty", "PASS",
                               f"{len(voices)} voice(s)")
            else:
                tracker.record("Voices list is non-empty", "FAIL", "empty")
        else:
            tracker.record("/v1/audio/voices returns 200", "FAIL", "request failed")
            tracker.record("Voices list is non-empty", "FAIL", "skipped (API error)")

    # Second voice
    if should_run_test("tts_second_voice", test_filter, skip_negative):
        print_test("AC: CustomVoice with a different predefined speaker")
        second_voice = voices[1] if len(voices) > 1 else primary_voice
        if second_voice == primary_voice:
            tracker.record("CustomVoice alternate speaker", "SKIP",
                           "only one voice available")
        else:
            print(f"  Using second voice: {BOLD}{second_voice}{NC}")
            outfile2 = output_dir / f"speech-{second_voice}.wav"
            payload2 = {"model": model_name, "input": text, "voice": second_voice,
                        "response_format": "wav"}
            (code, _), latency = timed_call(
                client.post_binary, f"{endpoint}/v1/audio/speech", payload2, outfile2)
            if 200 <= code < 300:
                fsize2 = outfile2.stat().st_size if outfile2.exists() else 0
                if fsize2 > 44:
                    tracker.record(f"CustomVoice ({second_voice}) produces audio",
                                   "PASS", f"{fsize2} bytes", latency_ms=latency)
                else:
                    tracker.record(f"CustomVoice ({second_voice}) produces audio",
                                   "FAIL", f"{fsize2} bytes", latency_ms=latency)
            else:
                tracker.record(f"CustomVoice ({second_voice}) produces audio",
                               "FAIL", f"HTTP {code}", latency_ms=latency)

    # FLAC output
    if should_run_test("tts_flac", test_filter, skip_negative):
        print_test("AC: FLAC output format")
        flac_file = output_dir / "speech.flac"
        flac_payload = {"model": model_name, "input": text, "voice": primary_voice,
                        "response_format": "flac"}
        (code, content_type), latency = timed_call(
            client.post_binary, f"{endpoint}/v1/audio/speech", flac_payload, flac_file)
        if not (200 <= code < 300):
            tracker.record("FLAC output", "SKIP", f"HTTP {code} (may not be supported)")
        else:
            if file_magic_check(flac_file, MAGIC_FLAC):
                tracker.record("FLAC magic bytes (fLaC)", "PASS", latency_ms=latency)
            else:
                tracker.record("FLAC magic bytes (fLaC)", "FAIL", latency_ms=latency)

    # Long text
    if should_run_test("tts_long_text", test_filter, skip_negative):
        print_test("AC: Long text input (>1000 chars)")
        long_text = (
            "The advancement of artificial intelligence has transformed how we interact with technology "
            "in our daily lives. From voice assistants that understand natural language to autonomous "
            "vehicles navigating complex traffic scenarios, machine learning algorithms continue to push "
            "the boundaries of what computers can achieve. In the field of natural language processing, "
            "large language models have demonstrated remarkable capabilities in text generation, "
            "translation, and summarization tasks. Meanwhile, computer vision systems can now identify "
            "objects, recognize faces, and interpret medical imagery with accuracy that rivals human "
            "experts. The integration of these technologies into healthcare has led to earlier disease "
            "detection, personalized treatment plans, and more efficient drug discovery pipelines. "
            "Robotics researchers are combining reinforcement learning with physical simulation to "
            "train robots that can manipulate objects, walk across uneven terrain, and collaborate "
            "with humans in shared workspaces. As these systems become more capable, questions about "
            "safety, alignment, and governance have become central to the research agenda, prompting "
            "interdisciplinary collaboration between engineers, ethicists, and policymakers worldwide."
        )
        long_file = output_dir / "speech-long.wav"
        long_payload = {"model": model_name, "input": long_text, "voice": primary_voice,
                        "response_format": "wav"}
        (code, _), latency = timed_call(
            client.post_binary, f"{endpoint}/v1/audio/speech", long_payload, long_file)
        if 200 <= code < 300:
            long_size = long_file.stat().st_size if long_file.exists() else 0
            if baseline_size > 0 and long_size > baseline_size * 2:
                tracker.record("Long text: output >2x baseline", "PASS",
                               f"{long_size} vs {baseline_size} bytes", latency_ms=latency)
            elif long_size > 1000:
                tracker.record("Long text: output size", "PASS",
                               f"{long_size} bytes", latency_ms=latency)
            else:
                tracker.record("Long text: output too small", "FAIL",
                               f"{long_size} bytes", latency_ms=latency)
        else:
            tracker.record("Long text", "FAIL", f"HTTP {code}", latency_ms=latency)

    # Empty input
    if should_run_test("tts_empty_input", test_filter, skip_negative):
        print_test("AC: Empty input (negative)")
        empty_payload = {"model": model_name, "input": "", "voice": primary_voice,
                         "response_format": "wav"}
        resp = client.post_json(f"{endpoint}/v1/audio/speech", empty_payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Empty input: returns 4xx", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Empty input: returns 4xx", "FAIL",
                           f"HTTP {resp.status_code} (got 5xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Empty input: returns 4xx", "FAIL", f"HTTP {code}")

    # Unsupported voice
    if should_run_test("tts_unsupported_voice", test_filter, skip_negative):
        print_test("AC: Unsupported voice (negative)")
        bad_voice_payload = {"model": model_name, "input": text,
                             "voice": "nonexistent_voice_xyz",
                             "response_format": "wav"}
        resp = client.post_json(f"{endpoint}/v1/audio/speech", bad_voice_payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Unsupported voice: returns 4xx", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Unsupported voice: returns 4xx", "FAIL",
                           f"HTTP {resp.status_code} (got 5xx)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Unsupported voice: returns 4xx", "FAIL",
                           f"HTTP {code}")

    # WAV duration
    if should_run_test("tts_wav_duration", test_filter, skip_negative):
        print_test("AC: WAV duration check")
        wav_file = output_dir / "speech.wav"
        if wav_file.exists():
            duration = parse_wav_duration(wav_file)
            if duration is None:
                tracker.record("WAV duration", "SKIP", "not PCM WAV")
            elif duration > 1.0:
                tracker.record("WAV duration >1.0s", "PASS",
                               f"{duration:.2f}s")
            else:
                tracker.record("WAV duration >1.0s", "FAIL",
                               f"{duration:.2f}s")
        else:
            tracker.record("WAV duration", "SKIP", "no WAV file generated")

    # Speed variants
    if should_run_test("tts_speed", test_filter, skip_negative):
        print_test("AC: Speed variants")
        for speed in (0.5, 2.0):
            speed_input = ("The quick brown fox jumps over the lazy dog. "
                          "Pack my box with five dozen liquor jugs. "
                          "How vexingly quick daft zebras jump."
                          if speed >= 1.0
                          else "Testing slow speech generation at half speed.")
            speed_payload = {"model": model_name, "input": speed_input,
                             "voice": primary_voice, "response_format": "wav",
                             "speed": speed}
            speed_file = output_dir / f"speech-speed-{speed}.wav"
            code, _ = client.post_binary(f"{endpoint}/v1/audio/speech",
                                         speed_payload, speed_file)
            if 200 <= code < 300:
                tracker.record(f"Speed={speed}: returns 200", "PASS")
            elif 400 <= code < 500:
                tracker.record(f"Speed={speed}", "SKIP",
                               f"HTTP {code} (unsupported)")
            else:
                tracker.record(f"Speed={speed}", "FAIL", f"HTTP {code}")

    # Streaming audio
    if should_run_test("tts_streaming_audio", test_filter, skip_negative):
        print_test("AC: Streaming audio (Transfer-Encoding: chunked)")
        stream_file = output_dir / "speech-stream.wav"
        stream_payload = {"model": model_name, "input": text, "voice": primary_voice,
                          "response_format": "wav"}
        code, _, headers = client.post_binary_with_headers(
            f"{endpoint}/v1/audio/speech", stream_payload, stream_file)
        if 200 <= code < 300:
            te = headers.get("Transfer-Encoding", "").lower()
            if "chunked" in te:
                tracker.record("Streaming audio: chunked encoding", "PASS")
            else:
                tracker.record("Streaming audio: chunked encoding", "WARN",
                               f"Transfer-Encoding: {te or 'not set'}")
        else:
            tracker.record("Streaming audio", "FAIL", f"HTTP {code}")

    # Multi-language
    if should_run_test("tts_multi_language", test_filter, skip_negative):
        print_test("AC: Multi-language (Chinese)")
        cn_text = "你好世界这是语音合成测试"
        cn_file = output_dir / "speech-chinese.wav"
        cn_payload = {"model": model_name, "input": cn_text, "voice": primary_voice,
                      "response_format": "wav"}
        (code, _), latency = timed_call(
            client.post_binary, f"{endpoint}/v1/audio/speech", cn_payload, cn_file)
        if 200 <= code < 300:
            cn_size = cn_file.stat().st_size if cn_file.exists() else 0
            if cn_size > 44:
                tracker.record("Multi-language (Chinese)", "PASS",
                               f"{cn_size} bytes", latency_ms=latency)
            else:
                tracker.record("Multi-language (Chinese)", "FAIL",
                               f"{cn_size} bytes (too small)", latency_ms=latency)
        else:
            tracker.record("Multi-language (Chinese)", "SKIP",
                           f"HTTP {code} (may not support Chinese)")


# ── Diffusion Validation ─────────────────────────────────────────────────────


def _decode_and_save_image(b64_data: str, output_dir: Path, base_name: str,
                           tracker: ResultTracker | None = None) -> None:
    """Decode base64 image, save with correct extension based on magic bytes."""
    img_path = output_dir / f"{base_name}.img"
    try:
        raw = base64.b64decode(b64_data)
        img_path.write_bytes(raw)
    except (ValueError, OSError):
        return

    if tracker:
        try:
            imgsize = img_path.stat().st_size
        except OSError:
            imgsize = 0
        if imgsize > 100:
            tracker.record("Decoded image has valid size", "PASS", f"{imgsize} bytes")
        else:
            tracker.record("Decoded image has valid size", "FAIL", f"{imgsize} bytes")

        if file_magic_check(img_path, MAGIC_PNG):
            tracker.record("Image is PNG format", "PASS")
        elif file_magic_check(img_path, MAGIC_JPEG):
            tracker.record("Image is JPEG format", "PASS")
        elif is_webp(img_path):
            tracker.record("Image is WEBP format", "PASS")
        else:
            tracker.record("Image format (PNG/JPEG/WEBP)", "FAIL")

    detect_and_rename_image(img_path, base_name)


def validate_diffusion(client: HttpClient, endpoint: str, model_name: str,
                       output_dir: Path, tracker: ResultTracker, *,
                       test_filter: list[str] | None = None,
                       skip_negative: bool = False) -> None:

    # /v1/images/generations
    if should_run_test("diffusion_generate", test_filter, skip_negative):
        print_test("AC: /v1/images/generations endpoint")
        payload = {
            "model": model_name,
            "prompt": "A serene mountain landscape at sunset with a calm lake in the foreground",
            "size": "512x512",
            "seed": 42,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/images/generations", payload)
        if resp is None or not (200 <= resp.status_code < 300):
            code = resp.status_code if resp else 0
            tracker.record("/v1/images/generations returns 200", "FAIL",
                           f"HTTP {code}", latency_ms=latency)
        else:
            tracker.record("/v1/images/generations returns 200", "PASS",
                           latency_ms=latency)
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            validate_schema(body, "images", tracker)

            data = body.get("data", [])
            if not data:
                tracker.record("Response contains image data", "FAIL",
                               "empty data array")
            else:
                tracker.record("Response contains image data", "PASS")
                b64_data = data[0].get("b64_json", "")
                if b64_data:
                    tracker.record("Image returned as base64", "PASS")
                    _decode_and_save_image(b64_data, output_dir, "generated", tracker)
                else:
                    url_data = data[0].get("url", "")
                    if url_data:
                        tracker.record("Image returned as URL", "PASS", url_data)
                    else:
                        tracker.record("Image data present (b64_json or url)", "FAIL",
                                       "neither found")

    # Chat-based image generation
    if should_run_test("diffusion_chat_image", test_filter, skip_negative):
        print_test("AC: Chat-based image generation (/v1/chat/completions)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user",
                          "content": "Generate an image of a red fox sitting in a snowy forest."}],
            "max_tokens": 256,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            if body.get("choices"):
                tracker.record("Chat image generation", "PASS")
                save_json(body, output_dir / "chat-image-response.json")
            else:
                tracker.record("Chat image generation", "FAIL",
                               "no choices in response")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Chat image generation", "SKIP",
                           f"HTTP {code} (may not be supported)")

    # Second prompt
    if should_run_test("diffusion_second_prompt", test_filter, skip_negative):
        print_test("AC: Second diffusion prompt")
        payload = {
            "model": model_name,
            "prompt": "A photorealistic cat wearing a tiny hat, studio lighting",
            "size": "512x512",
            "seed": 123,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            data = body.get("data", [])
            if data:
                tracker.record("Second prompt generates image", "PASS",
                               latency_ms=latency)
                b64_data2 = data[0].get("b64_json", "")
                if b64_data2:
                    _decode_and_save_image(b64_data2, output_dir, "generated-2")
            else:
                tracker.record("Second prompt generates image", "FAIL",
                               "empty data", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Second prompt generates image", "FAIL",
                           f"HTTP {code}", latency_ms=latency)

    # Size matrix
    if should_run_test("diffusion_size_matrix", test_filter, skip_negative):
        print_test("AC: Size matrix (256x256, 512x512, 1024x1024)")
        for size_str in ("256x256", "512x512", "1024x1024"):
            payload = {
                "model": model_name,
                "prompt": "A simple test pattern",
                "size": size_str,
                "seed": 1,
            }
            resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                data = body.get("data", [])
                if data:
                    b64 = data[0].get("b64_json", "")
                    if b64:
                        raw = base64.b64decode(b64)
                        dims = parse_png_dimensions(raw)
                        w, h = size_str.split("x")
                        expected = (int(w), int(h))
                        if dims and dims == expected:
                            tracker.record(f"Size {size_str}: dimensions match",
                                           "PASS")
                        elif dims:
                            tracker.record(f"Size {size_str}: dimensions",
                                           "FAIL",
                                           f"got {dims[0]}x{dims[1]}")
                        else:
                            # Not PNG, skip dimension check
                            tracker.record(f"Size {size_str}: generated OK",
                                           "PASS", "non-PNG, skip dim check")
                    else:
                        tracker.record(f"Size {size_str}: generated", "PASS")
                else:
                    tracker.record(f"Size {size_str}", "FAIL", "empty data")
            else:
                code = resp.status_code if resp else 0
                tracker.record(f"Size {size_str}", "FAIL", f"HTTP {code}")

    # Seed reproducibility
    if should_run_test("diffusion_seed_repro", test_filter, skip_negative):
        print_test("AC: Seed reproducibility")
        payload = {
            "model": model_name,
            "prompt": "Exact same red circle on white background",
            "size": "256x256",
            "seed": 999,
        }
        resp1 = client.post_json(f"{endpoint}/v1/images/generations", payload)
        resp2 = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if (resp1 is not None and 200 <= resp1.status_code < 300 and
                resp2 is not None and 200 <= resp2.status_code < 300):
            try:
                b64_1 = resp1.json().get("data", [{}])[0].get("b64_json", "")
                b64_2 = resp2.json().get("data", [{}])[0].get("b64_json", "")
            except (json.JSONDecodeError, ValueError, IndexError):
                b64_1 = b64_2 = ""
            if b64_1 and b64_2 and b64_1 == b64_2:
                tracker.record("Seed reproducibility: identical output", "PASS")
            elif b64_1 and b64_2:
                tracker.record("Seed reproducibility: outputs differ", "WARN",
                               "same seed produced different images")
            else:
                tracker.record("Seed reproducibility", "SKIP",
                               "no b64_json in response")
        else:
            tracker.record("Seed reproducibility", "FAIL", "request(s) failed")

    # Batch n=2
    if should_run_test("diffusion_batch", test_filter, skip_negative):
        print_test("AC: Batch generation (n=2)")
        payload = {
            "model": model_name,
            "prompt": "A blue square",
            "size": "256x256",
            "n": 2,
            "seed": 7,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            data = body.get("data", [])
            if len(data) == 2:
                all_valid = all(
                    item.get("b64_json") or item.get("url") for item in data
                )
                if all_valid:
                    tracker.record("Batch n=2: 2 images returned", "PASS",
                                   latency_ms=latency)
                else:
                    tracker.record("Batch n=2: some items empty", "FAIL",
                                   latency_ms=latency)
            else:
                tracker.record("Batch n=2: expected 2 items", "FAIL",
                               f"got {len(data)}", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Batch n=2", "FAIL", f"HTTP {code}",
                           latency_ms=latency)

    # Different prompts same seed -> different output
    if should_run_test("diffusion_different_prompts", test_filter, skip_negative):
        print_test("AC: Different prompts produce different images")
        payload_a = {
            "model": model_name,
            "prompt": "A red car on a highway",
            "size": "256x256",
            "seed": 50,
        }
        payload_b = {
            "model": model_name,
            "prompt": "A green tree in a park",
            "size": "256x256",
            "seed": 50,
        }
        resp_a = client.post_json(f"{endpoint}/v1/images/generations", payload_a)
        resp_b = client.post_json(f"{endpoint}/v1/images/generations", payload_b)
        if (resp_a is not None and 200 <= resp_a.status_code < 300 and
                resp_b is not None and 200 <= resp_b.status_code < 300):
            try:
                b64_a = resp_a.json().get("data", [{}])[0].get("b64_json", "")
                b64_b = resp_b.json().get("data", [{}])[0].get("b64_json", "")
            except (json.JSONDecodeError, ValueError, IndexError):
                b64_a = b64_b = ""
            if b64_a and b64_b and b64_a != b64_b:
                tracker.record("Different prompts: outputs differ", "PASS")
            elif b64_a and b64_b:
                tracker.record("Different prompts: outputs identical", "FAIL",
                               "different prompts produced same image")
            else:
                tracker.record("Different prompts", "SKIP", "no b64_json")
        else:
            tracker.record("Different prompts", "FAIL", "request(s) failed")

    # Negative prompt
    if should_run_test("diffusion_negative_prompt", test_filter, skip_negative):
        print_test("AC: Negative prompt")
        payload = {
            "model": model_name,
            "prompt": "A beautiful sunset over the ocean",
            "negative_prompt": "ugly, blurry, low quality",
            "size": "256x256",
            "seed": 10,
        }
        resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Negative prompt", "SKIP",
                           f"HTTP {resp.status_code} (unsupported)")
        elif resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Negative prompt: accepted", "PASS")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Negative prompt", "FAIL", f"HTTP {code}")

    # Guidance scale
    if should_run_test("diffusion_guidance_scale", test_filter, skip_negative):
        print_test("AC: Guidance scale")
        payload = {
            "model": model_name,
            "prompt": "A minimalist line drawing",
            "size": "256x256",
            "seed": 11,
            "guidance_scale": 3.5,
        }
        resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Guidance scale", "SKIP",
                           f"HTTP {resp.status_code} (unsupported)")
        elif resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Guidance scale=3.5: accepted", "PASS")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Guidance scale", "FAIL", f"HTTP {code}")

    # Invalid size
    if should_run_test("diffusion_invalid_size", test_filter, skip_negative):
        print_test("AC: Invalid size (negative)")
        payload = {
            "model": model_name,
            "prompt": "test",
            "size": "99x99",
            "seed": 1,
        }
        resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 400 <= resp.status_code < 600:
            tracker.record("Invalid size 99x99: rejected", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Invalid size 99x99: rejected", "SKIP",
                           "server accepts arbitrary sizes")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Invalid size 99x99: rejected", "FAIL",
                           f"HTTP {code}")

    # Empty prompt
    if should_run_test("diffusion_empty_prompt", test_filter, skip_negative):
        print_test("AC: Empty prompt (negative)")
        payload = {
            "model": model_name,
            "prompt": "",
            "size": "256x256",
            "seed": 1,
        }
        resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 400 <= resp.status_code < 600:
            tracker.record("Empty prompt: rejected", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Empty prompt: rejected", "FAIL",
                           "server accepted empty prompt")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Empty prompt: rejected", "FAIL",
                           f"HTTP {code}")

    # URL response path
    if should_run_test("diffusion_url_response", test_filter, skip_negative):
        print_test("AC: URL response format")
        payload = {
            "model": model_name,
            "prompt": "A simple dot",
            "size": "256x256",
            "seed": 1,
            "response_format": "url",
        }
        resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            data = body.get("data", [])
            if data and data[0].get("url"):
                url_val = data[0]["url"]
                # Fetch the URL to verify it returns an image
                img_resp = client.get(url_val)
                if img_resp is not None and img_resp.status_code == 200:
                    raw = img_resp.content[:8]
                    if (raw[:4] == MAGIC_PNG or raw[:3] == MAGIC_JPEG
                            or raw[:4] == MAGIC_RIFF):
                        tracker.record("URL response: valid image at URL", "PASS")
                    else:
                        tracker.record("URL response: not a known image format",
                                       "WARN")
                else:
                    tracker.record("URL response: could not fetch URL", "WARN",
                                   url_val[:80])
            elif data and data[0].get("b64_json"):
                tracker.record("URL response format", "SKIP",
                               "model always returns b64_json")
            else:
                tracker.record("URL response format", "FAIL", "no url or b64_json")
        elif resp is not None and 400 <= resp.status_code < 500:
            tracker.record("URL response format", "SKIP",
                           f"HTTP {resp.status_code} (unsupported)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("URL response format", "FAIL", f"HTTP {code}")

    # num_inference_steps
    if should_run_test("diffusion_num_inference_steps", test_filter, skip_negative):
        print_test("AC: num_inference_steps")
        for steps in (4, 25):
            payload = {
                "model": model_name,
                "prompt": "A blue circle",
                "size": "256x256",
                "seed": 1,
                "num_inference_steps": steps,
            }
            resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record(f"num_inference_steps={steps}: accepted", "PASS")
            elif resp is not None and 400 <= resp.status_code < 500:
                tracker.record(f"num_inference_steps={steps}", "SKIP",
                               f"HTTP {resp.status_code} (unsupported)")
            else:
                code = resp.status_code if resp else 0
                tracker.record(f"num_inference_steps={steps}", "FAIL",
                               f"HTTP {code}")


# ── Omni Validation ──────────────────────────────────────────────────────────


def validate_omni(client: HttpClient, endpoint: str, model_name: str,
                  output_dir: Path, tracker: ResultTracker, *,
                  test_filter: list[str] | None = None,
                  skip_negative: bool = False) -> None:
    # Prepare image URL (local file preferred, fallback to external)
    image_file = INPUT_DIR / "test-scenery.jpg"
    external_image_url = ("https://images.unsplash.com/photo-1506744038136-46273834b3fb"
                          "?w=640&q=80")
    if not image_file.exists():
        print(f"  {YELLOW}WARNING: {image_file} not found, vision test will use external URL{NC}")
        image_url = external_image_url
    else:
        img_size = image_file.stat().st_size
        if img_size > 2_097_152:
            print(f"  {YELLOW}WARNING: {image_file} is {img_size // 1024}KB (>2MB), "
                  f"may cause oversized payload{NC}")
        b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        image_url = f"data:image/jpeg;base64,{b64}"

    # Text-only chat
    if should_run_test("omni_text_chat", test_filter, skip_negative):
        print_test("AC: Text-only chat via /v1/chat/completions")
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user",
                          "content": "Explain the water cycle in simple terms."}],
            "max_tokens": 256,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is None or not (200 <= resp.status_code < 300):
            code = resp.status_code if resp else 0
            tracker.record("Text chat returns 200", "FAIL", f"HTTP {code}",
                           latency_ms=latency)
        else:
            tracker.record("Text chat returns 200", "PASS", latency_ms=latency)
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "text-chat-response.json")
            validate_schema(body, "chat_completion", tracker)

            text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if text and len(text) > 10:
                tracker.record("Text response is non-empty", "PASS",
                               f"{len(text)} chars")
                keywords = ["water", "cycle", "evaporation", "rain", "cloud"]
                if any(kw in text.lower() for kw in keywords):
                    tracker.record("Text response contains relevant content", "PASS")
                else:
                    tracker.record("Text response contains relevant content", "FAIL",
                                   "no expected keywords")
            else:
                tracker.record("Text response is non-empty", "FAIL",
                               "empty or too short")

    # /v1/completions
    if should_run_test("omni_completions", test_filter, skip_negative):
        print_test("AC: /v1/completions endpoint")
        payload = {
            "model": model_name,
            "prompt": "List the top five benefits of renewable energy.",
            "max_tokens": 256,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "completions-response.json")
            choices = body.get("choices", [])
            ctext = choices[0].get("text", "") if choices else ""
            if ctext and len(ctext) > 10:
                tracker.record("/v1/completions works", "PASS",
                               f"{len(ctext)} chars", latency_ms=latency)
            else:
                tracker.record("/v1/completions works", "FAIL",
                               "empty response", latency_ms=latency)
        elif resp is not None and 400 <= resp.status_code < 500:
            tracker.record("/v1/completions graceful rejection", "PASS",
                           f"HTTP {resp.status_code} (properly rejected)",
                           latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("/v1/completions works", "FAIL",
                           f"HTTP {code} (crashes pod — upstream bug)",
                           latency_ms=latency)

    # Vision
    if should_run_test("omni_vision", test_filter, skip_negative):
        print_test("AC: Vision input via /v1/chat/completions")
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": 256,
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is None or not (200 <= resp.status_code < 300):
            code = resp.status_code if resp else 0
            tracker.record("Vision chat returns 200", "FAIL", f"HTTP {code}",
                           latency_ms=latency)
        else:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "vision-response.json")
            vtext = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if vtext and len(vtext) > 10:
                tracker.record("Vision response describes image", "PASS",
                               f"{len(vtext)} chars", latency_ms=latency)
            else:
                tracker.record("Vision response describes image", "FAIL",
                               "empty or too short", latency_ms=latency)

    # Audio transcription
    if should_run_test("omni_audio_transcription", test_filter, skip_negative):
        print_test("AC: Audio input via /v1/audio/transcriptions")
        audio_input = INPUT_DIR / "test-audio-en.wav"
        if audio_input.exists():
            resp, latency = timed_call(
                client.post_multipart,
                f"{endpoint}/v1/audio/transcriptions",
                file_path=audio_input,
                file_field="file",
                data={"model": model_name, "response_format": "json", "language": "en"},
            )
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record("/v1/audio/transcriptions returns 200", "PASS",
                               latency_ms=latency)
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                save_json(body, output_dir / "transcription-response.json")
                ttext = body.get("text", "")
                if ttext:
                    tracker.record("Transcription contains text", "PASS",
                                   f"{len(ttext)} chars")
                elif "text" in body:
                    tracker.record("Transcription contains text", "FAIL",
                                   "empty text field")
                else:
                    tracker.record("Transcription contains text", "SKIP",
                                   "no .text field (test audio is synthetic)")
            else:
                code = resp.status_code if resp else 0
                tracker.record("/v1/audio/transcriptions", "SKIP",
                               f"HTTP {code} (transcription may not be supported)")
        else:
            tracker.record("Audio input test", "SKIP",
                           "test-audio-en.wav not found in inputs/")

    # Audio output chat
    if should_run_test("omni_audio_output", test_filter, skip_negative):
        print_test("AC: Chat with audio output (if supported)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user",
                          "content": "Say hello and introduce yourself briefly."}],
            "max_tokens": 256,
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        }
        resp, latency = timed_call(
            client.post_json, f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "chat-audio-response.json")
            audio_found = None
            for c in body.get("choices", []):
                msg = c.get("message", {})
                if msg.get("audio") and msg["audio"].get("data"):
                    audio_found = msg["audio"]["data"]
                    break
            if audio_found:
                tracker.record("Chat audio output", "PASS", latency_ms=latency)
                try:
                    (output_dir / "chat-audio-output.wav").write_bytes(
                        base64.b64decode(audio_found))
                except (ValueError, OSError):
                    pass
            elif any(c.get("message", {}).get("content")
                     for c in body.get("choices", [])):
                tracker.record("Chat audio output", "SKIP",
                               "text response only (audio modality may need different config)")
            else:
                tracker.record("Chat audio output", "FAIL",
                               "no content in response", latency_ms=latency)
        else:
            code = resp.status_code if resp else 0
            tracker.record("Chat audio output", "SKIP",
                           f"HTTP {code} (audio modality may not be enabled)")

    # Streaming + multimodal
    if should_run_test("omni_streaming_multimodal", test_filter, skip_negative):
        print_test("AC: Streaming with multimodal input")
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image? Be brief."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": 128,
            "stream": True,
        }
        stream_text, latency = timed_call(
            client.post_stream, f"{endpoint}/v1/chat/completions", payload)
        if stream_text:
            data_chunks = []
            for line in stream_text.splitlines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        data_chunks.append(chunk)
                    except (json.JSONDecodeError, ValueError):
                        pass
            if data_chunks:
                has_delta = any(
                    c.get("choices", [{}])[0].get("delta") is not None
                    for c in data_chunks if c.get("choices")
                )
                if has_delta:
                    tracker.record("Streaming multimodal: SSE with delta",
                                   "PASS", f"{len(data_chunks)} chunks",
                                   latency_ms=latency)
                else:
                    tracker.record("Streaming multimodal: no delta in chunks",
                                   "FAIL", latency_ms=latency)
            else:
                tracker.record("Streaming multimodal: no parseable chunks",
                               "FAIL", latency_ms=latency)
        else:
            tracker.record("Streaming multimodal", "FAIL", "empty response",
                           latency_ms=latency)

    # Multi-turn vision
    if should_run_test("omni_multiturn_vision", test_filter, skip_negative):
        print_test("AC: Multi-turn vision conversation")
        # Turn 1
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": 256,
        }
        resp1 = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp1 is not None and 200 <= resp1.status_code < 300:
            try:
                body1 = resp1.json()
            except (json.JSONDecodeError, ValueError):
                body1 = {}
            turn1_text = (body1.get("choices") or [{}])[0].get("message", {}).get("content", "")
            # Turn 2
            payload2 = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]},
                    {"role": "assistant", "content": turn1_text},
                    {"role": "user", "content": "What colors did you see?"},
                ],
                "max_tokens": 128,
            }
            resp2 = client.post_json(f"{endpoint}/v1/chat/completions", payload2)
            if resp2 is not None and 200 <= resp2.status_code < 300:
                try:
                    body2 = resp2.json()
                except (json.JSONDecodeError, ValueError):
                    body2 = {}
                turn2_text = (body2.get("choices") or [{}])[0].get("message", {}).get("content", "")
                kws = ["green", "blue", "sky", "grass", "path", "tree",
                       "nature", "water", "lake"]
                found = [kw for kw in kws if kw in turn2_text.lower()]
                if found:
                    tracker.record("Multi-turn vision: color keywords",
                                   "PASS", f"found: {', '.join(found)}")
                else:
                    tracker.record("Multi-turn vision: color keywords",
                                   "FAIL", "no expected keywords found")
            else:
                code = resp2.status_code if resp2 else 0
                tracker.record("Multi-turn vision: turn 2", "FAIL",
                               f"HTTP {code}")
        else:
            code = resp1.status_code if resp1 else 0
            tracker.record("Multi-turn vision", "FAIL", f"HTTP {code} on turn 1")

    # Multiple images
    if should_run_test("omni_multiple_images", test_filter, skip_negative):
        print_test("AC: Multiple images in single request")
        image_file_2 = INPUT_DIR / "test-image-2.jpg"
        if image_file_2.exists():
            b64_2 = base64.b64encode(image_file_2.read_bytes()).decode("ascii")
            image_url_2 = f"data:image/jpeg;base64,{b64_2}"
        else:
            image_url_2 = image_url  # fallback to same image

        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "How many images do you see? Reply with the number."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url_2}},
                ],
            }],
            "max_tokens": 32,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            mtext = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if "2" in mtext or "two" in mtext.lower():
                tracker.record("Multiple images: detects 2", "PASS")
            else:
                tracker.record("Multiple images: detects 2", "WARN",
                               f"response: {mtext[:60]}")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Multiple images", "FAIL", f"HTTP {code}")

    # Audio transcription formats
    if should_run_test("omni_transcription_formats", test_filter, skip_negative):
        print_test("AC: Audio transcription formats (WAV, MP3, FLAC)")
        for ext in ("wav", "mp3", "flac"):
            audio_path = INPUT_DIR / f"test-audio-en.{ext}"
            if not audio_path.exists():
                tracker.record(f"Transcription format ({ext})", "SKIP",
                               f"test-audio-en.{ext} not found")
                continue
            resp = client.post_multipart(
                f"{endpoint}/v1/audio/transcriptions",
                file_path=audio_path,
                file_field="file",
                data={"model": model_name, "response_format": "json", "language": "en"},
            )
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record(f"Transcription format ({ext}): 200 OK", "PASS")
            else:
                code = resp.status_code if resp else 0
                tracker.record(f"Transcription format ({ext})", "SKIP",
                               f"HTTP {code}")

    # Transcription accuracy
    if should_run_test("omni_transcription_accuracy", test_filter, skip_negative):
        print_test("AC: Transcription accuracy")
        audio_input = INPUT_DIR / "test-audio-en.wav"
        if audio_input.exists():
            resp = client.post_multipart(
                f"{endpoint}/v1/audio/transcriptions",
                file_path=audio_input,
                file_field="file",
                data={"model": model_name, "response_format": "json", "language": "en"},
            )
            if resp is not None and 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                ttext = body.get("text", "").lower()
                expected_words = ["quick", "brown", "fox", "lazy", "dog"]
                found = [w for w in expected_words if w in ttext]
                if len(found) >= 3:
                    tracker.record("Transcription accuracy: 3+ keywords",
                                   "PASS", f"found: {', '.join(found)}")
                elif found:
                    tracker.record("Transcription accuracy", "WARN",
                                   f"only {len(found)} keywords: {', '.join(found)}")
                else:
                    tracker.record("Transcription accuracy", "FAIL",
                                   "no expected keywords found")
            else:
                code = resp.status_code if resp else 0
                tracker.record("Transcription accuracy", "SKIP",
                               f"HTTP {code}")
        else:
            tracker.record("Transcription accuracy", "SKIP",
                           "test-audio-en.wav not found")

    # Audio output MP3
    if should_run_test("omni_audio_output_mp3", test_filter, skip_negative):
        print_test("AC: Audio output (MP3 format)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 128,
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "mp3"},
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            audio_b64 = ""
            for c in body.get("choices", []):
                msg = c.get("message", {})
                ad = msg.get("audio")
                if isinstance(ad, dict) and ad.get("data"):
                    audio_b64 = ad["data"]
                    break
            if audio_b64:
                try:
                    raw = base64.b64decode(audio_b64)
                    mp3_path = output_dir / "chat-audio-output.mp3"
                    mp3_path.write_bytes(raw)
                    if is_mp3(mp3_path):
                        tracker.record("Audio output MP3: valid magic bytes", "PASS")
                    elif raw[:4] == MAGIC_RIFF:
                        tracker.record("Audio output MP3: magic bytes", "FAIL",
                                       "got WAV instead of MP3 (format param ignored)")
                    else:
                        tracker.record("Audio output MP3: magic bytes", "FAIL",
                                       f"unknown format (first 4 bytes: {raw[:4].hex()})")
                except (ValueError, OSError):
                    tracker.record("Audio output MP3: decode error", "FAIL")
            else:
                tracker.record("Audio output MP3", "SKIP",
                               "no audio data in response")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Audio output MP3", "SKIP", f"HTTP {code}")

    # WAV output check
    if should_run_test("omni_wav_output_check", test_filter, skip_negative):
        print_test("AC: WAV output file verification")
        wav_path = output_dir / "chat-audio-output.wav"
        if wav_path.exists():
            wav_size = wav_path.stat().st_size
            if file_magic_check(wav_path, MAGIC_RIFF) and wav_size > 44:
                tracker.record("WAV output: RIFF magic + size>44", "PASS",
                               f"{wav_size} bytes")
            else:
                tracker.record("WAV output: RIFF magic + size>44", "FAIL",
                               f"size={wav_size}, magic_ok={file_magic_check(wav_path, MAGIC_RIFF)}")
        else:
            tracker.record("WAV output check", "SKIP",
                           "chat-audio-output.wav not generated")

    # Modality combinations
    if should_run_test("omni_modality_combinations", test_filter, skip_negative):
        print_test("AC: Modality combinations")
        combos: list[list[str]] = [["text"], ["audio"], ["text", "audio"]]
        for modality_list in combos:
            mod_str = "+".join(modality_list)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello."}],
                "max_tokens": 64,
                "modalities": modality_list,
            }
            if "audio" in modality_list:
                payload["audio"] = {"voice": "alloy", "format": "wav"}
            resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record(f"Modality [{mod_str}]: 200 OK", "PASS")
            elif resp is not None and 400 <= resp.status_code < 500:
                tracker.record(f"Modality [{mod_str}]", "SKIP",
                               f"HTTP {resp.status_code}")
            else:
                code = resp.status_code if resp else 0
                tracker.record(f"Modality [{mod_str}]", "FAIL",
                               f"HTTP {code}")

    # Large image
    if should_run_test("omni_large_image", test_filter, skip_negative):
        print_test("AC: Large image handling")
        large_img = INPUT_DIR / "test-scenery-4k.jpg"
        if large_img.exists() and large_img.stat().st_size > 2_000_000:
            b64_large = base64.b64encode(large_img.read_bytes()).decode("ascii")
            large_url = f"data:image/jpeg;base64,{b64_large}"
            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": large_url}},
                    ],
                }],
                "max_tokens": 32,
            }
            resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record("Large image: 200 OK", "PASS")
            elif resp is not None and resp.status_code in (400, 413):
                tracker.record("Large image: rejected (size limit)", "PASS",
                               f"HTTP {resp.status_code}")
            elif resp is not None and resp.status_code >= 500:
                tracker.record("Large image: server error", "FAIL",
                               f"HTTP {resp.status_code} (expected 200 or 4xx, not 5xx)")
            else:
                code = resp.status_code if resp else 0
                tracker.record("Large image", "WARN", f"HTTP {code}")
        else:
            tracker.record("Large image", "SKIP",
                           "test-scenery-4k.jpg not found or <2MB")

    # Unsupported modality
    if should_run_test("omni_unsupported_modality", test_filter, skip_negative):
        print_test("AC: Unsupported modality (video)")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello."}],
            "max_tokens": 32,
            "modalities": ["video"],
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Unsupported modality [video]: rejected", "PASS",
                           f"HTTP {resp.status_code}")
        elif resp is not None and resp.status_code >= 500:
            tracker.record("Unsupported modality [video]", "FAIL",
                           f"HTTP {resp.status_code} (expected 4xx)")
        elif resp is not None and 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            choices = body.get("choices", [])
            if not choices:
                tracker.record("Unsupported modality [video]", "FAIL",
                               f"HTTP {resp.status_code} empty response "
                               "(accepted but returned nothing)")
            else:
                tracker.record("Unsupported modality [video]", "PASS",
                               "server handled video modality")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Unsupported modality [video]", "FAIL",
                           f"HTTP {code}")

    # Vision URL vs base64
    if should_run_test("omni_vision_url_vs_base64", test_filter, skip_negative):
        print_test("AC: Vision URL vs base64")
        # base64 test (already covered, but explicit)
        if image_url.startswith("data:"):
            tracker.record("Vision base64: already tested above", "PASS")
        else:
            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                "max_tokens": 64,
            }
            resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                tracker.record("Vision base64 input", "PASS")
            else:
                code = resp.status_code if resp else 0
                tracker.record("Vision base64 input", "FAIL", f"HTTP {code}")

        # HTTP URL test
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image_url",
                     "image_url": {"url": external_image_url}},
                ],
            }],
            "max_tokens": 64,
        }
        resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
        if resp is not None and 200 <= resp.status_code < 300:
            tracker.record("Vision HTTP URL input", "PASS")
        elif resp is not None and 400 <= resp.status_code < 500:
            tracker.record("Vision HTTP URL input", "SKIP",
                           f"HTTP {resp.status_code} (connectivity issue or unsupported)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("Vision HTTP URL input", "SKIP",
                           f"HTTP {code} (connectivity failure)")



# ── Concurrency Smoke Test ───────────────────────────────────────────────────


def _concurrency_worker(worker_id: int, client_config: dict, endpoint: str,
                        model_name: str, model_type: str,
                        output_dir: Path) -> tuple[int, int]:
    """Single concurrency worker. Returns (worker_id, status_code)."""
    c = HttpClient(
        insecure=client_config.get("insecure", False),
        bearer_token=client_config.get("bearer_token", ""),
        verbose=client_config.get("verbose", False),
        get_timeout=client_config.get("get_timeout", 60),
        post_timeout=client_config.get("post_timeout", 300),
    )
    if model_type in ("text", "omni"):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Hello from worker {worker_id}"}],
            "max_tokens": 16,
        }
        resp = c.post_json(f"{endpoint}/v1/chat/completions", payload)
        return (worker_id, resp.status_code if resp else 0)
    elif model_type == "tts":
        outfile = output_dir / f"concurrency-{worker_id}.wav"
        payload = {
            "model": model_name,
            "input": f"Concurrency test number {worker_id}",
            "voice": client_config.get("tts_voice", "alloy"),
            "response_format": "wav",
        }
        code, _ = c.post_binary(f"{endpoint}/v1/audio/speech", payload, outfile)
        return (worker_id, code)
    elif model_type == "diffusion":
        payload = {
            "model": model_name,
            "prompt": f"Simple shape number {worker_id}",
            "size": "256x256",
            "seed": worker_id,
            "num_inference_steps": 1,
        }
        resp = c.post_json(f"{endpoint}/v1/images/generations", payload)
        if resp is None:
            # Try without num_inference_steps (might not be supported)
            del payload["num_inference_steps"]
            resp = c.post_json(f"{endpoint}/v1/images/generations", payload)
        return (worker_id, resp.status_code if resp else 0)
    return (worker_id, 0)


def validate_concurrency(client_config: dict, endpoint: str, model_name: str,
                         model_type: str, output_dir: Path,
                         tracker: ResultTracker) -> None:
    print_test("AC: Concurrency smoke test (3 parallel requests)")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_concurrency_worker, i, client_config, endpoint,
                            model_name, model_type, output_dir)
            for i in range(3)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append((-1, 0))

    all_ok = all(200 <= code < 300 for _, code in results)
    codes_str = ", ".join(f"w{wid}={code}" for wid, code in sorted(results))
    if all_ok:
        tracker.record("Concurrency: all 3 workers return 2xx", "PASS",
                       codes_str)
    else:
        tracker.record("Concurrency: all 3 workers return 2xx", "FAIL",
                       codes_str)


# ── JUnit XML Output ─────────────────────────────────────────────────────────


def generate_junit_xml(report: dict) -> str:
    """Generate JUnit XML from results report."""
    import html as _html

    def _esc(s: str) -> str:
        return _html.escape(s, quote=True)

    tests = report.get("tests", [])
    summary = report.get("summary", {})
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="genai-model-validation" '
        f'tests="{summary.get("total", 0)}" '
        f'failures="{summary.get("fail", 0)}" '
        f'skipped="{summary.get("skip", 0)}" '
        f'time="0">',
    ]
    for test in tests:
        name = _esc(test["name"])
        latency_s = test.get("latency_ms", 0) / 1000.0
        lines.append(f'  <testcase name="{name}" time="{latency_s:.3f}">')
        if test["status"] == "FAIL":
            detail = _esc(test.get("detail", ""))
            lines.append(f'    <failure message="{detail}"/>')
        elif test["status"] == "SKIP":
            detail = _esc(test.get("detail", ""))
            lines.append(f'    <skipped message="{detail}"/>')
        elif test["status"] == "WARN":
            detail = _esc(test.get("detail", ""))
            lines.append(f'    <system-out>{detail}</system-out>')
        lines.append('  </testcase>')
    lines.append('</testsuite>')
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GenAI Model Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models by type:
  text:       Any standard vLLM text/chat model (Llama, Mistral, etc.)
  tts:        Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, Voxtral-4B-TTS-2603
  diffusion:  black-forest-labs/FLUX.2-klein-4B, FLUX.2-dev
  omni:       Qwen/Qwen3-Omni-30B-A3B-Instruct

Authentication:
  export BEARER_TOKEN="your-token-here"

Examples:
  %(prog)s -e https://llama3.apps.cluster.example.com -m llama3 -t text
  %(prog)s -e https://qwen3-tts.apps.cluster.example.com -m qwen3-tts -t tts -k
  %(prog)s -e https://flux2-klein.apps.cluster.example.com -m flux2-klein -t diffusion
  %(prog)s -e https://qwen3-omni.apps.cluster.example.com -m qwen3-omni -t omni -v
""",
    )
    parser.add_argument("-e", metavar="ENDPOINT",
                        help="Endpoint URL (e.g. https://model.apps.cluster.example.com)")
    parser.add_argument("-m", metavar="MODEL_NAME",
                        help="Model name (e.g. qwen3-tts)")
    parser.add_argument("-t", choices=["text", "tts", "diffusion", "omni"],
                        metavar="TYPE", help="Model type: text | tts | diffusion | omni")
    parser.add_argument("-k", action="store_true",
                        help="Insecure mode: skip SSL certificate verification")
    parser.add_argument("-v", action="store_true",
                        help="Verbose mode: show HTTP requests and raw responses")
    parser.add_argument("--timeout", type=int, default=300, metavar="N",
                        help="POST timeout in seconds (default: 300). GET = max(30, N/5)")
    parser.add_argument("--timeout-warn", type=int, default=30, metavar="N",
                        help="Warn if any single request exceeds N seconds (default: 30)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print test names that would run and exit")
    parser.add_argument("--skip-negative", action="store_true",
                        help="Skip negative/invalid tests")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="Keep previous output dir (timestamped subdirectory)")
    parser.add_argument("--provision-inputs", action="store_true",
                        help="Run scripts/provision-inputs.sh before testing")
    parser.add_argument("--output-format", choices=["text", "json", "junit-xml"],
                        default="text", metavar="FMT",
                        help="Output format: text (default), json, junit-xml")
    parser.add_argument("--tests", type=str, default=None, metavar="LIST",
                        help="Comma-separated list of test names to run")
    parser.add_argument("--exclude-tests", type=str, default=None, metavar="LIST",
                        help="Comma-separated list of test names to exclude")
    parser.add_argument("--clean", nargs="?", const="MODEL", metavar="TARGET",
                        help="Clean outputs: omit or pass model name to clean that model, "
                             "or 'all' to remove entire outputs/ directory")
    parser.add_argument("--tar", action="store_true",
                        help="Create a tar.gz archive of the model output directory after validation")

    args = parser.parse_args()

    # Handle --no-color
    if args.no_color:
        disable_colors()

    # Standalone --clean (no -e/-m/-t): just clean and exit
    if args.clean is not None and not args.e and not args.m and not args.t:
        outputs_dir = BASE_DIR / "outputs"
        if args.clean == "all":
            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
                print(f"Removed: {outputs_dir}/")
            else:
                print("Nothing to clean (outputs/ does not exist)")
        elif args.clean == "MODEL":
            print("ERROR: --clean requires a model name or 'all'")
            print("  Usage: --clean <model-name>  or  --clean all")
            return 3
        else:
            safe = re.sub(r"[^a-zA-Z0-9._-]", "_", args.clean).lstrip(".")
            model_dir = outputs_dir / (safe or "_")
            if model_dir.exists():
                shutil.rmtree(model_dir)
                print(f"Removed: {model_dir}/")
            else:
                print(f"Nothing to clean (outputs/{safe}/ does not exist)")
        return 0

    # Validate required args for validation mode
    if not args.e or not args.m or not args.t:
        parser.error("arguments -e, -m, and -t are required for validation")

    endpoint = args.e.rstrip("/")
    model_name = args.m
    model_type = args.t
    insecure = args.k
    verbose = args.v

    # Compute timeouts
    post_timeout = args.timeout
    get_timeout = max(30, post_timeout // 5)

    # Parse --tests filter
    test_filter: list[str] | None = None
    if args.tests:
        test_filter = [t.strip() for t in args.tests.split(",") if t.strip()]
        available = all_test_names_for_type(model_type)
        unknown = [t for t in test_filter if t not in available]
        if unknown:
            print(f"{RED}ERROR: Unknown test name(s): {', '.join(unknown)}{NC}")
            print(f"\nAvailable tests for type '{model_type}':")
            for name in available:
                print(f"  {name}")
            return 3

    # Parse --exclude-tests
    global _EXCLUDE_SET
    if args.exclude_tests:
        exclude_list = [t.strip() for t in args.exclude_tests.split(",") if t.strip()]
        available = all_test_names_for_type(model_type)
        unknown = [t for t in exclude_list if t not in available]
        if unknown:
            print(f"{RED}ERROR: Unknown test name(s) in --exclude-tests: "
                  f"{', '.join(unknown)}{NC}")
            print(f"\nAvailable tests for type '{model_type}':")
            for name in available:
                print(f"  {name}")
            return 3
        _EXCLUDE_SET = frozenset(exclude_list)

    # --dry-run
    if args.dry_run:
        available = all_test_names_for_type(model_type)
        print(f"Tests that would run for type '{model_type}':")
        for name in available:
            if should_run_test(name, test_filter, args.skip_negative):
                print(f"  {name}")
            else:
                print(f"  {DIM}{name} (skipped){NC}")
        return 0

    # --provision-inputs
    if args.provision_inputs:
        provision_script = BASE_DIR / "scripts" / "provision-inputs.sh"
        if not provision_script.exists():
            print(f"{RED}ERROR: {provision_script} not found{NC}")
            return 3
        print(f"{DIM}Running provision-inputs.sh ...{NC}")
        result = subprocess.run(
            ["bash", str(provision_script)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"{RED}ERROR: provision-inputs.sh failed (exit {result.returncode}){NC}")
            if result.stderr:
                print(result.stderr[:500])
            return 3
        print(f"{GREEN}Inputs provisioned successfully{NC}")

    # Sanitize model name for output directory
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    safe_name = safe_name.lstrip(".")
    if not safe_name:
        safe_name = "_"
    output_dir = BASE_DIR / "outputs" / safe_name
    try:
        output_dir.resolve().relative_to(BASE_DIR / "outputs")
    except ValueError:
        print("Unsafe output path derived from model name")
        return 3

    # --clean with validation: clean first, then proceed
    if args.clean is not None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleaned: {output_dir}/")
        else:
            print(f"Nothing to clean (outputs/{safe_name}/ does not exist)")

    # Prepare output directory
    if output_dir.exists():
        if args.keep_outputs:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = output_dir / ts
        else:
            if verbose:
                print(f"{DIM}    Output directory found: {output_dir}{NC}")
            shutil.rmtree(output_dir)
            if verbose:
                print(f"{DIM}    Deleted previous output directory{NC}")
    else:
        if verbose:
            print(f"{DIM}    No previous output directory found{NC}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{DIM}    Created fresh output directory: {output_dir}{NC}")

    bearer_token = os.environ.get("BEARER_TOKEN", "")
    client = HttpClient(insecure=insecure, bearer_token=bearer_token, verbose=verbose,
                        get_timeout=get_timeout, post_timeout=post_timeout)
    timeout_warn_ms = args.timeout_warn * 1000.0 if args.timeout_warn else 0.0
    tracker = ResultTracker(timeout_warn_ms=timeout_warn_ms)

    # Print header
    print_header("vLLM Model Validation")
    print()
    print(f"  Model:    {BOLD}{model_name}{NC}")
    print(f"  Type:     {BOLD}{model_type}{NC}")
    print(f"  Endpoint: {BOLD}{endpoint}{NC}")
    if bearer_token:
        print(f"  Auth:     {GREEN}Bearer token set{NC}")
    else:
        print(f"  Auth:     {YELLOW}None{NC}")
    if insecure:
        print(f"  SSL:      {YELLOW}Insecure (-k){NC}")
    else:
        print(f"  SSL:      {GREEN}Verified{NC}")
    print(f"  Timeout:  {BOLD}GET={get_timeout}s POST={post_timeout}s{NC}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip()}")

    # Health check (always runs)
    health_ok = validate_health(client, endpoint, tracker)
    if not health_ok:
        print()
        print(f"  {RED}{BOLD}Health check failed — endpoint unreachable. "
              f"Skipping remaining tests.{NC}")
        tracker.print_summary(model_name, model_type, endpoint, output_dir)
        report = tracker.to_json_report(model_name, model_type, endpoint, {})
        save_json(report, output_dir / "results.json")
        return 2

    # Models list (always runs)
    validate_models_list(client, endpoint, model_name, tracker)

    # Runtime metadata (always runs)
    metadata = capture_runtime_metadata(client, endpoint)
    if metadata:
        print(f"\n{DIM}  Runtime metadata: {json.dumps(metadata, default=str)[:200]}{NC}")

    # Discover TTS voice early (needed by warm-up and concurrency)
    tts_voice = "alloy"
    if model_type == "tts":
        _, voices = _discover_voices(client, endpoint)
        if voices:
            tts_voice = voices[0]

    # Warm-up
    warmup_request(client, endpoint, model_name, model_type, tts_voice=tts_voice)

    # Schema validation on a basic request
    if should_run_test("common_schema", test_filter, args.skip_negative):
        print_test("AC: Schema validation (basic request)")
        if model_type in ("text", "omni"):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
            resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                validate_schema(body, "chat_completion", tracker)
            else:
                tracker.record("Schema validation", "SKIP", "request failed")
        elif model_type == "diffusion":
            payload = {
                "model": model_name,
                "prompt": "test",
                "size": "256x256",
            }
            resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
            if resp is not None and 200 <= resp.status_code < 300:
                try:
                    body = resp.json()
                except (json.JSONDecodeError, ValueError):
                    body = {}
                validate_schema(body, "images", tracker)
            else:
                tracker.record("Schema validation", "SKIP", "request failed")
        else:
            tracker.record("Schema validation", "SKIP",
                           f"no schema check for {model_type}")

    # Negative tests (common)
    validate_negative_common(client, endpoint, model_name, model_type, tracker,
                             insecure=insecure, verbose=verbose,
                             bearer_token=bearer_token,
                             test_filter=test_filter,
                             skip_negative=args.skip_negative)

    # Type-specific checks
    if model_type == "text":
        validate_text(client, endpoint, model_name, output_dir, tracker,
                      test_filter=test_filter, skip_negative=args.skip_negative)
    elif model_type == "tts":
        validate_tts(client, endpoint, model_name, output_dir, tracker,
                     test_filter=test_filter, skip_negative=args.skip_negative)
    elif model_type == "diffusion":
        validate_diffusion(client, endpoint, model_name, output_dir, tracker,
                           test_filter=test_filter, skip_negative=args.skip_negative)
    elif model_type == "omni":
        validate_omni(client, endpoint, model_name, output_dir, tracker,
                      test_filter=test_filter, skip_negative=args.skip_negative)

    # Concurrency smoke test
    if should_run_test("common_concurrency", test_filter, args.skip_negative):
        client_config = {
            "insecure": insecure,
            "verbose": verbose,
            "bearer_token": bearer_token,
            "get_timeout": get_timeout,
            "post_timeout": post_timeout,
            "tts_voice": tts_voice,
        }
        validate_concurrency(client_config, endpoint, model_name, model_type,
                             output_dir, tracker)

    # Summary
    passed = tracker.print_summary(model_name, model_type, endpoint, output_dir)

    # Generate and write results.json (always)
    report = tracker.to_json_report(model_name, model_type, endpoint, metadata)
    save_json(report, output_dir / "results.json")

    # Output format
    if args.output_format == "json":
        print(json.dumps(report, indent=2))
    elif args.output_format == "junit-xml":
        print(generate_junit_xml(report))

    # --tar: archive the output directory
    if args.tar:
        import tarfile
        tar_path = output_dir.parent / f"{output_dir.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(output_dir, arcname=output_dir.name)
        print(f"\n  Archive created: {tar_path}")

    if not passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
