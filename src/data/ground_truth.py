from pathlib import Path
import re


def load_transcription(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # split into lines, strip empty and comment lines
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        lines.append(line)
    return lines


def load_all_transcriptions(gt_dir):
    gt_dir = Path(gt_dir)
    transcriptions = {}
    for f in sorted(gt_dir.iterdir()):
        if f.suffix in (".txt", ".md"):
            source_name = f.stem
            transcriptions[source_name] = load_transcription(f)
            print(f"  {source_name}: {len(transcriptions[source_name])} lines")
    return transcriptions


def build_line_pairs(transcription_lines, line_image_paths):
    """Pair ground truth lines with detected line images.
    Assumes line_image_paths are sorted in reading order and
    len(transcription_lines) <= len(line_image_paths)."""
    n = min(len(transcription_lines), len(line_image_paths))
    pairs = []
    for i in range(n):
        pairs.append({
            "image_path": str(line_image_paths[i]),
            "text": transcription_lines[i],
        })
    return pairs
