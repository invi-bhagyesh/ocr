import os
from abc import ABC, abstractmethod
from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def query(self, image_path, prompt, temperature=0.1):
        pass

    @abstractmethod
    def query_text_only(self, prompt, temperature=0.1):
        pass

    def query_with_few_shot(self, image_path, prompt, examples, temperature=0.1):
        """Default: prepend example texts to prompt. Subclasses can override
        to send actual image+text pairs as multimodal few-shot."""
        if not examples:
            return self.query(image_path, prompt, temperature)
        shots = "Examples from this document:\n"
        for i, ex in enumerate(examples, 1):
            shots += f"  {i}. \"{ex['text']}\"\n"
        return self.query(image_path, shots + "\n" + prompt, temperature)


class QwenLocalClient(VLMClient):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 adapter_path=None, load_in_4bit=True):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        )

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()

    def _generate(self, messages, temperature=0.1, max_tokens=2048):
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )
        generated = output_ids[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def query(self, image_path, prompt, temperature=0.1):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]}]
        return self._generate(messages, temperature).strip()

    def query_text_only(self, prompt, temperature=0.1):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}]
        return self._generate(messages, temperature).strip()

    def query_with_few_shot(self, image_path, prompt, examples, temperature=0.1):
        """True multimodal few-shot: each example is an image+text turn."""
        if not examples:
            return self.query(image_path, prompt, temperature)

        messages = []
        for ex in examples:
            # each example becomes a user→assistant turn with the line image
            if "image_path" in ex:
                messages.append({"role": "user", "content": [
                    {"type": "image", "image": str(ex["image_path"])},
                    {"type": "text", "text": prompt},
                ]})
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": ex["text"]},
                ]})
            else:
                # text-only example as fallback
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Example transcription: \"{ex['text']}\""},
                ]})
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": "Understood."},
                ]})

        # final turn: the actual image to transcribe
        messages.append({"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]})
        return self._generate(messages, temperature).strip()


class GeminiClient(VLMClient):
    """Zero-shot baseline using Gemini API."""

    def __init__(self, model="gemini-2.0-flash", api_key=None):
        import google.generativeai as genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def query(self, image_path, prompt, temperature=0.1):
        img = Image.open(image_path)
        response = self.model.generate_content(
            [prompt, img],
            generation_config={"temperature": temperature, "max_output_tokens": 2048}
        )
        return response.text.strip()

    def query_text_only(self, prompt, temperature=0.1):
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 2048}
        )
        return response.text.strip()

    def query_with_few_shot(self, image_path, prompt, examples, temperature=0.1):
        """Gemini multimodal few-shot: interleave example images and texts."""
        if not examples:
            return self.query(image_path, prompt, temperature)

        parts = [prompt + "\n\nExamples:\n"]
        for i, ex in enumerate(examples, 1):
            if "image_path" in ex:
                parts.append(Image.open(ex["image_path"]))
            parts.append(f"Transcription {i}: \"{ex['text']}\"\n")

        parts.append("\nNow transcribe this image:\n")
        parts.append(Image.open(image_path))

        response = self.model.generate_content(
            parts,
            generation_config={"temperature": temperature, "max_output_tokens": 2048}
        )
        return response.text.strip()


def get_client(backend="qwen", **kwargs):
    if backend == "qwen":
        return QwenLocalClient(**kwargs)
    elif backend == "gemini":
        return GeminiClient(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
