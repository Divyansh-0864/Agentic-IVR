#!/usr/bin/env python3
"""
whisper_benchmark.py

Benchmark OpenAI's whisper (PyTorch) vs faster-whisper (CTranslate2) on CPU.
Measures: load_time, memory (RSS) snapshots, transcribe_time, WER.
Output: bench_results.csv

Usage examples:
  # manifest.csv must have header: audio_path,reference
  python whisper_benchmark.py --manifest data/manifest.csv --models "whisper:tiny,faster-whisper:tiny" --compute_type int8 --cpu_threads 4

  # single file:
  python whisper_benchmark.py --file examples/audio1.wav --ref "expected transcription" --models "whisper:tiny,faster-whisper:tiny"
"""
import argparse
import csv
import time
import os
import sys
import psutil
import gc
import re
from typing import List, Tuple
import pandas as pd

# Basic WER (token-level Levenshtein)
def wer(ref: str, hyp: str) -> float:
    def normalize(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)
        return s.split()
    r = normalize(ref)
    h = normalize(hyp)
    # dp (len(r)+1 x len(h)+1)
    n, m = len(r), len(h)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    edit = dp[n][m]
    return edit / n

def sample_rss_mb():
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    return rss / (1024*1024)

def prepare_audio_list(args):
    audios = []
    if args.manifest:
        df = pd.read_csv(args.manifest)
        if 'audio_path' not in df.columns or 'reference' not in df.columns:
            raise ValueError("manifest CSV must contain columns: audio_path,reference")
        for _, row in df.iterrows():
            audios.append((str(row['audio_path']), str(row['reference'])))
    else:
        if not args.file or args.ref is None:
            raise ValueError("Either --manifest or both --file and --ref must be provided")
        audios.append((args.file, args.ref))
    # check files exist
    for p,_ in audios:
        if not os.path.exists(p):
            raise FileNotFoundError(f"audio file not found: {p}")
    return audios

def run_bench_for_whisper(model_name, audios, args):
    import whisper
    results = []
    gc.collect()
    before_rss = sample_rss_mb()
    t0 = time.perf_counter()
    model = whisper.load_model(model_name, device="cpu")
    load_time = time.perf_counter() - t0
    after_load_rss = sample_rss_mb()
    for audio_path, ref in audios:
        t0 = time.perf_counter()
        # whisper.model.transcribe returns dict with 'text'
        out = model.transcribe(audio_path, language=args.language if args.language else None, verbose=False)
        trans_time = time.perf_counter() - t0
        after_trans_rss = sample_rss_mb()
        text = out.get("text", "").strip()
        w = wer(ref, text)
        results.append({
            "framework": "whisper",
            "model": model_name,
            "audio": audio_path,
            "reference": ref,
            "hypothesis": text,
            "load_time_s": load_time,
            "rss_before_load_mb": before_rss,
            "rss_after_load_mb": after_load_rss,
            "transcribe_time_s": trans_time,
            "rss_after_transcribe_mb": after_trans_rss,
            "wer": w
        })
    return results

def run_bench_for_faster_whisper(model_name, audios, args):
    # faster-whisper API
    from faster_whisper import WhisperModel
    results = []
    gc.collect()
    before_rss = sample_rss_mb()
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type='int8', cpu_threads=args.cpu_threads)
    load_time = time.perf_counter() - t0
    after_load_rss = sample_rss_mb()
    for audio_path, ref in audios:
        t0 = time.perf_counter()
        # model.transcribe returns segments and info (sequential)
        segments, info = model.transcribe(audio_path, beam_size=args.beam_size, language=args.language if args.language else None)
        trans_time = time.perf_counter() - t0
        after_trans_rss = sample_rss_mb()
        # join segments
        text = "".join([s.text for s in segments]).strip()
        w = wer(ref, text)
        results.append({
            "framework": "faster-whisper",
            "model": model_name,
            "audio": audio_path,
            "reference": ref,
            "hypothesis": text,
            "load_time_s": load_time,
            "rss_before_load_mb": before_rss,
            "rss_after_load_mb": after_load_rss,
            "transcribe_time_s": trans_time,
            "rss_after_transcribe_mb": after_trans_rss,
            "wer": w
        })
    return results

# Call from dir = experiments/whisper
def main():
    parser = argparse.ArgumentParser(description="Benchmark whisper vs faster-whisper on CPU")
    parser.add_argument("--manifest", default="data/manifest.csv",help="CSV with header audio_path,reference (paths must be accessible)")
    parser.add_argument("--file", help="single audio file")
    parser.add_argument("--ref", help="reference transcription for single file")
    parser.add_argument("--models", required=True, help='Comma-separated list of models to test, format "<framework>:<model_name>". e.g. "whisper:tiny,faster-whisper:tiny"')
    parser.add_argument("--out", default="results/bench_results_small.csv", help="output CSV file")
    parser.add_argument("--language", default=None, help="language code to pass to transcribe() (optional)")
    parser.add_argument("--compute_type", default=None, help="compute_type for faster-whisper (e.g. int8, int16). If omitted, default is None")
    parser.add_argument("--cpu_threads", type=int, default=4, help="cpu threads for faster-whisper")
    parser.add_argument("--beam_size", type=int, default=5, help="beam size for faster-whisper (affects accuracy & speed)")
    args = parser.parse_args()

    audios = prepare_audio_list(args)
    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    all_results = []
    for spec in specs:
        if ":" not in spec:
            print(f"Bad model spec: {spec}. Use <framework>:<model_name> format, e.g. whisper:tiny", file=sys.stderr)
            continue
        fw, model_name = spec.split(":", 1)
        print(f"\n=== Running {fw} model '{model_name}' ===")
        try:
            if fw == "whisper":
                r = run_bench_for_whisper(model_name, audios, args)
            elif fw in ("faster-whisper", "faster_whisper", "fasterwhisper"):
                r = run_bench_for_faster_whisper(model_name, audios, args)
            else:
                print(f"Unknown framework '{fw}' — skipping", file=sys.stderr)
                continue
            all_results.extend(r)
        except Exception as e:
            print(f"Error running {fw}:{model_name} -> {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            # try to free memory between models
            gc.collect()

    if not all_results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")

    # Print summary per model
    summary = df.groupby(["framework","model"]).agg({
        "load_time_s":"first",
        "rss_after_load_mb":"mean",
        "transcribe_time_s":"mean",
        "rss_after_transcribe_mb":"mean",
        "wer":"mean"
    }).reset_index()
    print("\nSummary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()