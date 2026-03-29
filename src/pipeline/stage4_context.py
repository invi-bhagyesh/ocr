from pathlib import Path


def _load_prompt():
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "stage4_context.txt"
    return prompt_path.read_text()


def contextual_correction(page_lines, vlm_client):
    text_block = "\n".join(page_lines)
    prompt = _load_prompt().replace("{text}", text_block)
    response = vlm_client.query_text_only(prompt).strip()
    return response.split("\n")
