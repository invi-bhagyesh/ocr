from pathlib import Path


def _load_prompt():
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "stage2_transcribe.txt"
    return prompt_path.read_text()


def transcribe_line(line_image_path, vlm_client, few_shot_examples=None):
    prompt = _load_prompt()
    return vlm_client.query_with_few_shot(
        line_image_path, prompt, few_shot_examples or []
    ).strip()


def transcribe_page(line_image_paths, vlm_client, few_shot_examples=None):
    return [
        transcribe_line(path, vlm_client, few_shot_examples)
        for path in line_image_paths
    ]
