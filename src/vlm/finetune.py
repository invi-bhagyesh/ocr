import torch
from pathlib import Path
from PIL import Image
import random

TRANSCRIBE_PROMPT = (
    "Transcribe the handwritten text in this image exactly as written. "
    "Preserve original spelling — do not modernize. Output only the text."
)


def prepare_lora_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct", r=16, alpha=32):
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")
    return model


def augment_image(image):
    from torchvision import transforms
    aug = transforms.Compose([
        transforms.RandomRotation(2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ])
    return aug(image)


class OCRFineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, processor, augment=True):
        self.pairs = pairs
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")

        if self.augment and random.random() > 0.3:
            image = augment_image(image)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TRANSCRIBE_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": pair["text"]},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # mask everything before the assistant response so we only train on output
        assistant_token = self.processor.tokenizer.encode(
            "assistant", add_special_tokens=False
        )
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i+len(assistant_token)].tolist() == assistant_token:
                labels[:i + len(assistant_token)] = -100
                break

        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = labels
        return result


def collate_fn(batch, pad_id=0):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        tensors = [b[key] for b in batch]
        if key == "labels":
            max_len = max(t.size(0) for t in tensors)
            collated[key] = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, t in enumerate(tensors):
                collated[key][i, :t.size(0)] = t
        elif key == "pixel_values":
            collated[key] = torch.cat(tensors, dim=0)
        elif key == "image_grid_thw":
            collated[key] = torch.cat(tensors, dim=0)
        else:
            max_len = max(t.size(0) for t in tensors)
            collated[key] = torch.full((len(batch), max_len), pad_id, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                collated[key][i, :t.size(0)] = t
    return collated


def train_lora(model, processor, train_pairs, val_pairs=None, output_dir="checkpoints",
               epochs=5, lr=2e-4, batch_size=2, grad_accum=8):
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OCRFineTuneDataset(train_pairs, processor, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer.pad_token_id),
        drop_last=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    total_steps = epochs * len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    model.train()
    history = {"train_loss": []}
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            epoch_loss += outputs.loss.item()

        avg = epoch_loss / len(train_loader)
        history["train_loss"].append(avg)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

        model.save_pretrained(output_dir / f"epoch_{epoch+1}")

    model.save_pretrained(output_dir / "final")
    return output_dir / "final", history
