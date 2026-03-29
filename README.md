# RenAIssance OCR3 — VLM Pipeline for Handwritten Early Modern Spanish OCR

GSoC 2026 evaluation task for [HumanAI / RenAIssance](https://humanai.foundation/gsoc/2026/proposal_OCR3.html).

End-to-end OCR pipeline for 16th-17th century Spanish handwritten documents, using a Vision-Language Model at every stage — not just as a post-correction step.

## Task Completed

- **Test II**: VLM-based OCR pipeline for handwritten sources with 4-stage architecture, per-source evaluation, and per-stage ablation

## Repository Structure

```
renna/
├── src/
│   ├── data/
│   │   ├── pdf_convert.py     # PDF → JPEG page images at 300 DPI
│   │   ├── ground_truth.py    # load and align transcription files
│   │   └── dataset.py         # line-image + text pairs for finetuning
│   ├── pipeline/
│   │   ├── stage1_layout.py   # VLM-based text line detection
│   │   ├── stage2_transcribe.py  # VLM few-shot transcription
│   │   ├── stage3_correct.py  # multimodal self-correction (image + text)
│   │   ├── stage4_context.py  # LLM contextual text-only correction
│   │   └── runner.py          # pipeline orchestration with ablation support
│   ├── vlm/
│   │   ├── client.py          # VLM client abstraction (Gemini API / Qwen local)
│   │   └── finetune.py        # LoRA finetuning for Qwen2.5-VL
│   ├── eval/
│   │   └── metrics.py         # CER, WER, Normalized Levenshtein Similarity
│   └── utils/
│       └── image.py           # crop, resize, contrast enhancement, base64 encoding
├── prompts/
│   ├── stage1_layout.txt      # line detection prompt
│   ├── stage2_transcribe.txt  # transcription prompt
│   ├── stage3_correct.txt     # self-correction prompt
│   └── stage4_context.txt     # contextual correction prompt
├── notebooks/
│   └── 01_ocr_pipeline.ipynb  # full pipeline demo + evaluation + ablation
├── proposal/
│   └── proposal.md
├── data/
│   ├── raw_pdfs/              # place downloaded handwritten PDFs here
│   ├── pages/                 # generated JPEG pages
│   ├── lines/                 # cropped line images
│   └── ground_truth/          # place transcription files here
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

**Data**: Download the handwritten source PDFs and transcription files from the SharePoint links provided in the task description. Place PDFs in `data/raw_pdfs/` and transcriptions in `data/ground_truth/`.

**API key**: Set the `GEMINI_API_KEY` environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Running

```
notebooks/01_ocr_pipeline.ipynb
```

The notebook runs the full pipeline:

1. Converts PDFs to page images
2. Demonstrates the 4-stage pipeline on a single page (with line crop visualization)
3. Evaluates across all sources with ground truth
4. Per-stage ablation showing incremental CER improvement

## Pipeline Architecture

```
Page Image
    │
    ▼
[Stage 1: Layout Detection] ── VLM identifies text line bounding boxes
    │
    ▼
[Stage 2: Transcription] ── VLM few-shot OCR per cropped line
    │
    ▼
[Stage 3: Self-Correction] ── VLM re-reads image + Stage 2 output, fixes errors
    │
    ▼
[Stage 4: Context Correction] ── LLM text-only pass over full page
    │
    ▼
Final Transcription
```

The VLM is used at every stage, not just cleanup. Stage 3 (multimodal self-correction) is the key contribution — feeding both the line image and the initial transcription back to the VLM to catch misread characters.

## Evaluation Metrics

| Metric | Description                                                      |
| ------ | ---------------------------------------------------------------- |
| CER    | Character Error Rate — edit distance / reference length          |
| WER    | Word Error Rate — word-level edit distance / word count          |
| NLS    | Normalized Levenshtein Similarity — 1 - (edit_dist / max_length) |

Results are reported per source (each handwritten PDF) since handwriting styles vary. The ablation shows CER after each pipeline stage to quantify incremental improvement.

## VLM Backends

The pipeline supports swappable backends via `src/vlm/client.py`:

- **Gemini API** (default) — `gemini-2.0-flash`, free tier, strong multimodal reasoning
- **Qwen2.5-VL local** — open-source, supports LoRA finetuning on ground truth line-image pairs

## References

- Greif et al. (2025) — Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents
- Kim et al. (2025) — Early Evidence of How LLMs Outperform Traditional Systems on OCR/HTR Tasks for Historical Records
- Chung & Choi (2025) — Finetuning Vision-Language Models as OCR Systems for Low-Resource Languages
- Murrieta-Flores et al. (2025) — Unlocking Colonial Records with Artificial Intelligence
- Heidenreich et al. (2026) — GutenOCR: A Grounded Vision-Language Front-End for Documents
