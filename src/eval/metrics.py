import re
import editdistance


def cer(prediction, reference):
    if not reference:
        return 0.0 if not prediction else 1.0
    return editdistance.eval(prediction, reference) / len(reference)


def wer(prediction, reference):
    pred_words = prediction.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return editdistance.eval(pred_words, ref_words) / len(ref_words)


def normalized_levenshtein(prediction, reference):
    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 1.0
    return 1.0 - editdistance.eval(prediction, reference) / max_len


# early modern → modern Spanish normalization pairs
# if the model outputs the modern form when the reference has the historical form,
# that's a normalization error (data corruption for historians)
NORMALIZATION_PAIRS = [
    (r'\bdixo\b', 'dijo'), (r'\bvno\b', 'uno'), (r'\bvna\b', 'una'),
    (r'\bhazer\b', 'hacer'), (r'\bhizo\b', 'hizo'), (r'\bSeuilla\b', 'Sevilla'),
    (r'\bciudad\b', 'ciudad'), (r'\bescriuano\b', 'escribano'),
    (r'\bescriptura\b', 'escritura'), (r'\breçibir\b', 'recibir'),
    (r'\bmerçed\b', 'merced'), (r'\bdiçho\b', 'dicho'),
    (r'\bveynte\b', 'veinte'), (r'\bquatro\b', 'cuatro'),
]


def count_normalizations(prediction, reference):
    """Count instances where the model modernized a historical spelling
    that appears in the reference. These are harmful 'corrections'."""
    count = 0
    for historical_pattern, modern_form in NORMALIZATION_PAIRS:
        # reference has the historical form
        ref_matches = len(re.findall(historical_pattern, reference, re.IGNORECASE))
        if ref_matches == 0:
            continue
        # prediction has the modern form instead
        pred_has_modern = len(re.findall(r'\b' + re.escape(modern_form) + r'\b',
                                         prediction, re.IGNORECASE))
        pred_has_historical = len(re.findall(historical_pattern, prediction,
                                             re.IGNORECASE))
        # normalization = reference had historical, prediction replaced with modern
        if pred_has_modern > 0 and pred_has_historical < ref_matches:
            count += min(ref_matches - pred_has_historical, pred_has_modern)
    return count


def evaluate_lines(predictions, references):
    assert len(predictions) == len(references)
    n = len(predictions)
    total_cer = sum(cer(p, r) for p, r in zip(predictions, references))
    total_wer = sum(wer(p, r) for p, r in zip(predictions, references))
    total_nls = sum(normalized_levenshtein(p, r) for p, r in zip(predictions, references))
    total_norm = sum(count_normalizations(p, r) for p, r in zip(predictions, references))
    return {
        "cer": total_cer / n,
        "wer": total_wer / n,
        "nls": total_nls / n,
        "normalization_errors": total_norm,
        "n_lines": n,
    }


def evaluate_document(page_predictions, page_references):
    per_page = {}
    all_preds, all_refs = [], []
    for page_id in page_predictions:
        preds = page_predictions[page_id]
        refs = page_references.get(page_id, [])
        n = min(len(preds), len(refs))
        if n == 0:
            continue
        per_page[page_id] = evaluate_lines(preds[:n], refs[:n])
        all_preds.extend(preds[:n])
        all_refs.extend(refs[:n])

    overall = evaluate_lines(all_preds, all_refs) if all_preds else {}
    return {"per_page": per_page, "overall": overall}


def evaluate_ablation(stage_outputs, references):
    results = {}
    n = min(len(references), len(stage_outputs.get("stage2_raw", [])))
    if n == 0:
        return results
    refs = references[:n]

    for stage_key, label in [
        ("stage2_raw", "Stage 2: Raw Transcription"),
        ("stage3_corrected", "Stage 3: + Self-Correction"),
        ("stage4_final", "Stage 4: + Context Correction"),
    ]:
        preds = stage_outputs.get(stage_key, [])[:n]
        if preds:
            results[label] = evaluate_lines(preds, refs)
    return results
