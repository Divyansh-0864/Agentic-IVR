#!/usr/bin/env python3
"""
stt.py

Simple CLI STT runner supporting:
 - OpenAI whisper (PyTorch) via `whisper`
 - faster-whisper (CTranslate2) via `faster_whisper`

Usage examples:
  # transcribe single file using faster-whisper tiny (int8 quantized)
  python stt.py --backend faster-whisper --model tiny --compute_type int8 samples/customer_support.wav

  # transcribe multiple files and write CSV results
  python stt.py --backend whisper --model tiny --out results/whisper_tiny_results.csv samples/*.wav

Notes:
 - Ensure ffmpeg is installed and on PATH for whisper audio handling.
 - On low-RAM CPU, prefer tiny/base models and faster-whisper with compute_type=int8.

Run:
# Using faster-whisper
python services/stt.py experiments/whisper/data/samples/ivr_demo.mp3 --backend faster-whisper --model tiny --compute_type int8 --cpu_threads 4 --out experiments/whisper/results/bench_results_tiny.csv

# USing openAI
python services/stt.py experiments/whisper/data/samples/ivr_demo.mp3 --backend whisper --model tiny --out experiments/whisper/results/ivr_demo.csv

"""
import argparse
import os
import sys
import time
import psutil
import csv
from pathlib import Path
from typing import Optional, List, Dict

def sample_rss_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def timer() -> float:
    return time.perf_counter()

def check_ffmpeg_warn():
    import shutil
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found on PATH. Whisper may fail to load or transcode audio. Install ffmpeg.", file=sys.stderr)

class BaseSTT:
    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
    def load(self) -> Dict:
        raise NotImplementedError
    def transcribe(self, audio_path: str, language: Optional[str]=None, **kwargs) -> Dict:
        raise NotImplementedError

class WhisperSTT(BaseSTT):
    def __init__(self, model_name="tiny", device="cpu"):
        super().__init__(model_name)
        self.device = device
        self.model = None

    def load(self):
        check_ffmpeg_warn()
        import whisper
        t0 = timer()
        self.model = whisper.load_model(self.model_name, device=self.device)
        load_time = timer() - t0
        return {"framework": "whisper", "model": self.model_name, "load_time_s": load_time, "rss_after_load_mb": sample_rss_mb()}

    def transcribe(self, audio_path: str, language: Optional[str] = None, verbose: bool=False, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        t0 = timer()
        out = self.model.transcribe(audio_path, language=language, verbose=verbose)
        trans_time = timer() - t0
        text = out.get("text", "").strip()
        return {"text": text, "transcribe_time_s": trans_time, "rss_after_transcribe_mb": sample_rss_mb()}

class FasterWhisperSTT(BaseSTT):
    def __init__(self, model_name="tiny", device="cpu", compute_type: Optional[str]=None, cpu_threads: int = 4):
        super().__init__(model_name)
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.model = None

    def load(self):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError("Install faster-whisper (pip install faster-whisper). Error: " + str(e))
        t0 = timer()
        self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type, cpu_threads=self.cpu_threads)
        load_time = timer() - t0
        return {"framework": "faster-whisper", "model": self.model_name, "load_time_s": load_time, "rss_after_load_mb": sample_rss_mb()}

    def transcribe(self, audio_path: str, language: Optional[str] = None, beam_size: int = 5, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        t0 = timer()
        segments, info = self.model.transcribe(audio_path, beam_size=beam_size, language=language)
        trans_time = timer() - t0
        text = "".join([s.text for s in segments]).strip()
        return {"text": text, "transcribe_time_s": trans_time, "rss_after_transcribe_mb": sample_rss_mb(), "info": info}

def get_backend(name: str, model: str, compute_type: Optional[str], cpu_threads: int):
    name = name.lower()
    if name == "whisper":
        return WhisperSTT(model_name=model)
    elif name in ("faster-whisper", "faster_whisper", "fasterwhisper"):
        return FasterWhisperSTT(model_name=model, compute_type=compute_type, cpu_threads=cpu_threads)
    else:
        raise ValueError("Unknown backend: " + name)

def transcribe_files(backend_name: str, model: str, files: List[str], compute_type: Optional[str], cpu_threads: int, language: Optional[str], beam_size: int, out_csv: Optional[str]):
    be = get_backend(backend_name, model, compute_type, cpu_threads)
    load_metrics = be.load()
    results = []
    for f in files:
        fpath = Path(f)
        if not fpath.exists():
            print(f"File not found: {f}", file=sys.stderr)
            continue
        before_rss = sample_rss_mb()
        res = be.transcribe(str(fpath), language=language, beam_size=beam_size)
        after_rss = sample_rss_mb()
        row = {
            "framework": load_metrics.get("framework"),
            "model": load_metrics.get("model"),
            "audio": str(fpath),
            "text": res.get("text", ""),
            "load_time_s": load_metrics.get("load_time_s"),
            "rss_before_load_mb": None,                # not sampled before load here
            "rss_after_load_mb": load_metrics.get("rss_after_load_mb"),
            "transcribe_time_s": res.get("transcribe_time_s"),
            "rss_after_transcribe_mb": res.get("rss_after_transcribe_mb"),
        }
        results.append(row)
        print(f"[{row['framework']}:{row['model']}] {f} -> {len(row['text'])} chars, transcribe_time={row['transcribe_time_s']:.3f}s, rss={row['rss_after_transcribe_mb']:.1f}MB")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = ["framework","model","audio","text","load_time_s","rss_after_load_mb","transcribe_time_s","rss_after_transcribe_mb"]
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved results to {out_csv}")
    return results

def main():
    p = argparse.ArgumentParser(description="STT CLI: whisper / faster-whisper")
    p.add_argument("files", nargs="+", help="Audio files to transcribe (wav/mp3)")
    p.add_argument("--backend", default="faster-whisper", help="whisper | faster-whisper")
    p.add_argument("--model", default="tiny", help="model name (tiny, base, small, etc.)")
    p.add_argument("--compute_type", default=None, help="faster-whisper compute_type (int8, int16, etc.)")
    p.add_argument("--cpu_threads", type=int, default=4, help="cpu threads for faster-whisper")
    p.add_argument("--language", default=None, help="language code (optional)")
    p.add_argument("--beam_size", type=int, default=5, help="beam size for faster-whisper")
    p.add_argument("--out", default=None, help="Optional CSV path to save results")
    args = p.parse_args()

    # If whisper backend used, check ffmpeg
    if args.backend.lower() == "whisper":
        check_ffmpeg_warn()

    transcribe_files(backend_name=args.backend, model=args.model, files=args.files,
                     compute_type=args.compute_type, cpu_threads=args.cpu_threads,
                     language=args.language, beam_size=args.beam_size, out_csv=args.out)

if __name__ == "__main__":
    main()
