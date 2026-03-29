import json
import re
from pathlib import Path
from PIL import Image
from ..utils.image import crop_region, detect_lines_projection


def _load_prompt():
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "stage1_layout.txt"
    return prompt_path.read_text()


def detect_lines_vlm(page_image_path, vlm_client):
    """Pure VLM line detection — ask the model for bounding boxes."""
    prompt = _load_prompt()
    response = vlm_client.query(page_image_path, prompt)

    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        lines = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            lines = json.loads(match.group())
        else:
            return []

    return [entry["bbox"] for entry in lines if "bbox" in entry]


def detect_lines_hybrid(page_image_path, vlm_client):
    """Hybrid: projection profile for line boundaries, VLM to filter marginalia."""
    img = Image.open(page_image_path)
    w, h = img.size

    # get line y-ranges from projection profile
    y_ranges = detect_lines_projection(img)
    if not y_ranges:
        # fallback to pure VLM
        return detect_lines_vlm(page_image_path, vlm_client)

    # convert to full-width bboxes
    candidate_bboxes = [[0, y1, w, y2] for y1, y2 in y_ranges]

    # ask VLM to filter: which are main text vs marginalia?
    filter_prompt = (
        f"This page has {len(candidate_bboxes)} detected text regions. "
        f"Looking at the page image, which regions contain main body text "
        f"(not marginalia, not page numbers, not decorative elements)? "
        f"Return a JSON array of indices (0-based) for main text regions only."
    )
    response = vlm_client.query(page_image_path, filter_prompt)

    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        indices = json.loads(text)
        if isinstance(indices, list):
            return [candidate_bboxes[i] for i in indices
                    if isinstance(i, int) and 0 <= i < len(candidate_bboxes)]
    except (json.JSONDecodeError, IndexError):
        pass

    # if VLM filtering fails, return all candidates (better than nothing)
    return candidate_bboxes


def detect_lines(page_image_path, vlm_client, method="hybrid"):
    if method == "hybrid":
        return detect_lines_hybrid(page_image_path, vlm_client)
    return detect_lines_vlm(page_image_path, vlm_client)


def crop_lines(page_image_path, bboxes, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(page_image_path)

    paths = []
    for i, bbox in enumerate(bboxes):
        cropped = crop_region(img, bbox)
        out = output_dir / f"line_{i:03d}.jpg"
        cropped.save(str(out), "JPEG", quality=95)
        paths.append(out)
    return paths
