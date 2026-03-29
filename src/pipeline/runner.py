from pathlib import Path
from . import stage1_layout, stage2_transcribe, stage3_correct, stage4_context


class OCRPipeline:
    def __init__(self, vlm_client, few_shot_examples=None, lines_dir="data/lines"):
        self.client = vlm_client
        self.few_shot = few_shot_examples
        self.lines_dir = Path(lines_dir)

    def process_page(self, page_image_path, page_id="page"):
        line_dir = self.lines_dir / page_id
        bboxes = stage1_layout.detect_lines(page_image_path, self.client)
        line_paths = stage1_layout.crop_lines(page_image_path, bboxes, line_dir)

        raw_texts = stage2_transcribe.transcribe_page(
            line_paths, self.client, self.few_shot
        )
        corrected = stage3_correct.correct_page(line_paths, raw_texts, self.client)
        final = stage4_context.contextual_correction(corrected, self.client)

        return final

    def process_page_ablation(self, page_image_path, page_id="page"):
        """Run pipeline returning intermediate results at each stage."""
        line_dir = self.lines_dir / page_id
        bboxes = stage1_layout.detect_lines(page_image_path, self.client)
        line_paths = stage1_layout.crop_lines(page_image_path, bboxes, line_dir)

        stage2_out = stage2_transcribe.transcribe_page(
            line_paths, self.client, self.few_shot
        )
        stage3_out = stage3_correct.correct_page(line_paths, stage2_out, self.client)
        stage4_out = stage4_context.contextual_correction(stage3_out, self.client)

        return {
            "bboxes": bboxes,
            "line_paths": line_paths,
            "stage2_raw": stage2_out,
            "stage3_corrected": stage3_out,
            "stage4_final": stage4_out,
        }
