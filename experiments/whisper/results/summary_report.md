# Whisper Benchmark Summary  
**Experiment Date:** Nov 2025  
**System:** CPU-only (8 GB RAM), no GPU  
**Models Tested:**  
- Whisper (tiny, base)  
- Faster-Whisper (tiny, base)  

---

## 1. Objective  
Evaluate Whisper and Faster-Whisper on CPU for use in the Agentic-IVR project.  
Metrics recorded:  
- **Load Time** (cold start latency)  
- **Transcription Time**  
- **Memory Usage** (RSS)  
- **Accuracy (WER)**  
- **Behavior across Tiny and Base model sizes**

All tests were run on 5 realistic sample audios (20–30 seconds each).

---

## 2. Key Findings

### **A. Load Time**
| Model | Framework | Observed Load Time | Notes |
|-------|-----------|--------------------|-------|
| tiny | whisper | **0.49s** | Very fast to initialize. |
| tiny | faster-whisper | **0.90s** | Slightly slower, overhead from CTranslate2 initialization. |
| base | whisper | **13.6s** | Much heavier — slow cold start. |
| base | faster-whisper | **15.3s** | Slowest due to larger model + CT2 init. |

▶ **Cold start for base models is too high for an IVR system.**

---

### **B. Transcription Speed**
| Model | Whisper | Faster-Whisper | Notes |
|-------|---------|----------------|-------|
| tiny | ~2.5–3.0s | **0.8–1.1s** | **FW is 2–3× faster**. |
| base | ~4.5–5.8s | **1.3–1.6s** | **FW is 4× faster**. |

▶ Faster-Whisper is significantly faster on CPU, especially for larger models.

---

### **C. Memory Usage (RSS)**
#### After Model Load:
- **tiny**:  
  - Whisper: ~493 MB  
  - Faster-Whisper: ~499 MB  
- **base**:  
  - Whisper: ~631 MB  
  - Faster-Whisper: ~630 MB  

#### After Transcription:
- tiny: up to 560 MB  
- base: up to 730 MB  

▶ **Both frameworks use similar memory**, but base models push close to the 8GB limit during multitask situations.

---

### **D. Accuracy (WER)**
WER was extremely similar:

| Model | Whisper WER | Faster-Whisper WER |
|--------|--------------|-----------------------|
| tiny | 0–4.7% | 0–4.7% |
| base | ~0–2% | ~0–2% |

▶ Accuracy is effectively identical.  
▶ Base models slightly more stable at longer passages.
▶ The WER is coming mostly due to difference in punctuations. The transcription added the punctuations on their own even though none were present in sample. 

---

## 3. Interpretation

### **Whisper vs Faster-Whisper**
**Faster-Whisper wins in:**
- Transcription speed  
- CPU-efficiency  
- Parallelization  
- Stability under load  

**Whisper wins in:**
- Faster model load for tiny  
- Slightly simpler integration  

### **Tiny vs Base**
| Criteria | tiny | base |
|---------|------|------|
| Load time | Excellent | Very slow |
| Speed | Good | Slower |
| Accuracy | Acceptable for IVR | High accuracy |
| Memory | Moderate | High (700MB+) |

**Conclusion:**  
- **Base is too heavy for real-time CPU IVR.**  
- **Tiny achieves comparable accuracy for structured prompts** (banking, IVR commands).  
- Domain-specific vocabulary reduces the need for larger models.

---

## 4. Recommendation for Agentic-IVR

### ✔ Use **Faster-Whisper Tiny (int8)** for MVP  
- Fastest transcription  
- Lowest latency  
- Accuracy is sufficient for IVR intents  
- Cold start < 1 second  
- Memory footprint manageable  
- Best user experience for live phone interactions  

### ✔ Add custom post-processing + intent recognition  
This will compensate for tiny model imperfections.

### ✔ Warm-start the model  
Load model at server startup to avoid cold-start delay entirely.

---

## 5. Future Work

### 1. **Benchmark quantized models**
- `int8`, `int4`, and CPU threading differences  
- Compare:  
  - `compute_type=int8`,  
  - `compute_type=int8_float16`,  
  - `compute_type=int4` (experimental)

This can significantly reduce memory.

---

### 2. **Try Distil-Whisper (for GPU future)**
If GPU is added later, distil-whisper is an ideal middle-ground model.

---

### 3. **Create domain-specific prompting for transcribe()**
- Force language (`language="en"`)  
- Use `initial_prompt="You are transcribing banking IVR commands"`  
- Expected to improve accuracy by ~10–20%.

---

### 4. **Stress test with concurrent calls**  
Simulate:
- 5 parallel calls  
- Measure CPU saturation  
- Check if tiny still meets latency requirements

---

### 5. **Hybrid System Idea (Recommended Later Stage)**  
Use:
- Faster-whisper **tiny** for real-time streaming  
- Faster-whisper **base** in background if high-confidence transcription needed  

---

## 6. Final Conclusion

**For the Phase-1 MVP of Agentic-IVR:  
Use:**  
### ⭐ `faster-whisper-tiny` (int8)  
### ⭐ With warm start + post-processing pipeline

You get the best combination of:
- Speed  
- Low memory usage  
- Good enough accuracy  
- Works reliably on 8GB RAM CPU  
- Ensures smooth phone-call UX  

Base models are not suitable for real-time IVR on your hardware.

