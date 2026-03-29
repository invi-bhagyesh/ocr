from pathlib import Path


def _load_prompt():
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "stage3_correct.txt"
    return prompt_path.read_text()


def correct_line(line_image_path, raw_text, vlm_client):
    prompt = _load_prompt().replace("{text}", raw_text)
    return vlm_client.query(line_image_path, prompt).strip()


def correct_page(line_image_paths, raw_texts, vlm_client):
    results = []
    for path, text in zip(line_image_paths, raw_texts):
        corrected = correct_line(path, text, vlm_client)
        results.append(corrected)
    return results
