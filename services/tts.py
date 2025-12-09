#!/usr/bin/env python3
"""
tts.py

Simple CLI TTS tool using gTTS (Google Text-to-Speech).
Outputs MP3 files by default. Optionally converts to WAV if pydub and ffmpeg available.

Usage examples:
  # synthesize a short text to mp3
  python tts.py --text "Hello from Agentic IVR" --out_dir experiments/whisper/data/tts

  # synthesize multiple texts from a text file (one per line)
  python tts.py --input_file texts.txt --out_dir experiments/whisper/data/tts

Notes:
 - gTTS requires network access (it uses Google TTS).
 - To convert to WAV automatically, install pydub and ffmpeg.

Run:
python services/tts.py --input_file experiments/whisper/data/samples/ivr_demo.txt --out_dir experiments/whisper/data/samples

"""
import argparse
from gtts import gTTS
from pathlib import Path
import uuid
import sys

def synthesize_text(text: str, out_dir: str, filename: str = None, lang: str = "en", to_wav: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or f"tts_{uuid.uuid4().hex[:8]}.mp3"
    mp3_path = out_dir / filename
    tts = gTTS(text=text, lang=lang)
    tts.save(str(mp3_path))
    print(f"Saved MP3: {mp3_path}")
    if to_wav:
        # try to convert to WAV using pydub (requires ffmpeg)
        try:
            from pydub import AudioSegment
            wav_name = mp3_path.with_suffix(".wav")
            AudioSegment.from_file(mp3_path).export(wav_name, format="wav")
            print(f"Converted to WAV: {wav_name}")
            return str(wav_name)
        except Exception as e:
            print("WAV conversion failed (pydub/ffmpeg required):", e, file=sys.stderr)
    return str(mp3_path)

def synthesize_from_file(input_file: str, out_dir: str, lang: str = "en", to_wav: bool = False):
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(input_file)
    results = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            fname = f"tts_line_{i+1}.mp3"
            path = synthesize_text(text, out_dir, filename=fname, lang=lang, to_wav=to_wav)
            results.append(path)
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple gTTS CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to synthesize (wrap in quotes)")
    group.add_argument("--input_file", help="Text file with one utterance per line")
    parser.add_argument("--out_dir", default="experiments/whisper/data/tts", help="Output directory")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--to_wav", action="store_true", help="Also convert to WAV (requires pydub + ffmpeg)")
    args = parser.parse_args()

    if args.text:
        synthesize_text(args.text, args.out_dir, lang=args.lang, to_wav=args.to_wav)
    else:
        synthesize_from_file(args.input_file, args.out_dir, lang=args.lang, to_wav=args.to_wav)

if __name__ == "__main__":
    main()
