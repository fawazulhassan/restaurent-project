from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import config
from app.stt import SttInitError, preload_stt, transcribe_file_timed
from app.text_fold import compare_tokens


def _normalized(text: str) -> str:
    return " ".join(compare_tokens(text))


def _edit_distance(left: str, right: str) -> int:
    a = _normalized(left)
    b = _normalized(right)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _vram_used_mb() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


class VramSampler:
    def __init__(self) -> None:
        self.baseline = _vram_used_mb()
        self.max_used = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = _vram_used_mb()
            if used is not None and (self.max_used is None or used > self.max_used):
                self.max_used = used
            time.sleep(0.2)

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def delta(self) -> int | str:
        if self.baseline is None or self.max_used is None:
            return "n/a"
        return max(self.max_used - self.baseline, 0)


def _result_name() -> str:
    model = config.WHISPER_MODEL.replace("/", "-")
    device = config.WHISPER_DEVICE
    compute = config.WHISPER_COMPUTE_TYPE
    return f"benchmark-results-{model}-{device}-{compute}.json"


def run_benchmark(directory: Path, runs: int, warmup: int) -> int:
    expected_path = directory / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    device = config.WHISPER_DEVICE.strip().lower()

    sampler = VramSampler() if device != "cpu" else None
    context = sampler if sampler is not None else _NullContext()
    cuda_error_count = 0
    runs_completed = 0
    per_clip: dict[str, dict[str, Any]] = {}

    try:
        with context:
            preload_stt()
            for clip, expected_text in expected.items():
                path = directory / clip
                for _ in range(warmup):
                    try:
                        transcribe_file_timed(path, roman_bias=True, latinize=False)
                    except Exception:
                        if device != "cpu":
                            cuda_error_count += 1
                        break

                timings: list[float] = []
                transcript = ""
                distance = 0
                exact = False
                for _ in range(runs):
                    try:
                        transcript, seconds = transcribe_file_timed(
                            path,
                            roman_bias=True,
                            latinize=False,
                        )
                    except Exception:
                        if device != "cpu":
                            cuda_error_count += 1
                        continue
                    timings.append(seconds)
                    runs_completed += 1
                    distance = _edit_distance(transcript, expected_text)
                    exact = distance == 0

                per_clip[clip] = {
                    "expected": expected_text,
                    "transcript": transcript,
                    "distance": distance,
                    "exact_match": exact,
                    "median_seconds": statistics.median(timings) if timings else None,
                    "runs_completed": len(timings),
                }
    except SttInitError as exc:
        print(exc)
        return 1

    medians = [
        item["median_seconds"]
        for item in per_clip.values()
        if item["median_seconds"] is not None
    ]
    result = {
        "model": config.WHISPER_MODEL,
        "device": config.WHISPER_DEVICE,
        "compute_type": config.WHISPER_COMPUTE_TYPE,
        "runs_requested": runs,
        "warmup_requested": warmup,
        "cuda_error_count": cuda_error_count,
        "runs_completed": runs_completed,
        "per_clip_median_seconds": {
            clip: data["median_seconds"] for clip, data in per_clip.items()
        },
        "overall_median_seconds": statistics.median(medians) if medians else None,
        "clips": per_clip,
        "peak_vram_delta_mb": "n/a" if sampler is None else sampler.delta(),
    }
    out = Path(_result_name())
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def compare_results(cpu_path: Path, gpu_path: Path) -> int:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    cpu_clips = set(cpu["clips"])
    gpu_clips = set(gpu["clips"])
    if cpu_clips != gpu_clips:
        print("Benchmark compare error: clip sets do not match.")
        return 1
    if cpu.get("runs_requested") != gpu.get("runs_requested"):
        print("Benchmark compare error: run counts do not match.")
        return 1

    speed_ok = (
        gpu.get("overall_median_seconds") is not None
        and cpu.get("overall_median_seconds") is not None
        and gpu["overall_median_seconds"] <= cpu["overall_median_seconds"] * 0.7
    )
    cuda_ok = gpu.get("cuda_error_count") == 0
    quality_ok = all(
        gpu["clips"][clip]["distance"] <= cpu["clips"][clip]["distance"]
        for clip in cpu_clips
    )
    vram_value = gpu.get("peak_vram_delta_mb")
    vram_ok = isinstance(vram_value, (int, float)) and vram_value <= 3276.8
    decision = "ADOPT" if speed_ok and cuda_ok and quality_ok and vram_ok else "KEEP CPU"

    lines = [
        "# STT Benchmark Comparison",
        "",
        f"Decision: **{decision}**",
        "",
        f"CPU overall median: {cpu.get('overall_median_seconds')}",
        f"GPU overall median: {gpu.get('overall_median_seconds')}",
        f"GPU CUDA errors: {gpu.get('cuda_error_count')}",
        f"GPU peak VRAM delta MB: {gpu.get('peak_vram_delta_mb')}",
        "",
        "| Clip | CPU distance | GPU distance | GPU transcript |",
        "|---|---:|---:|---|",
    ]
    for clip in sorted(cpu_clips):
        lines.append(
            f"| {clip} | {cpu['clips'][clip]['distance']} | "
            f"{gpu['clips'][clip]['distance']} | {gpu['clips'][clip]['transcript']} |"
        )
    Path("benchmark-results.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local Whisper STT")
    parser.add_argument("--dir", default="data/test_audio")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare", nargs=2, metavar=("CPU_JSON", "GPU_JSON"))
    args = parser.parse_args()

    if args.compare:
        raise SystemExit(compare_results(Path(args.compare[0]), Path(args.compare[1])))
    raise SystemExit(run_benchmark(Path(args.dir), args.runs, args.warmup))


if __name__ == "__main__":
    main()
