from pathlib import Path
from pdf2image import convert_from_path


def pdf_to_images(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path), dpi=dpi)
    paths = []
    for i, img in enumerate(images):
        out = output_dir / f"{pdf_path.stem}_page_{i:03d}.jpg"
        img.save(str(out), "JPEG", quality=95)
        paths.append(out)
    return paths


def convert_all(pdf_dir, output_dir, dpi=300):
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    all_paths = {}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        source_dir = output_dir / pdf.stem
        pages = pdf_to_images(pdf, source_dir, dpi)
        all_paths[pdf.stem] = pages
        print(f"  {pdf.name}: {len(pages)} pages")
    return all_paths
