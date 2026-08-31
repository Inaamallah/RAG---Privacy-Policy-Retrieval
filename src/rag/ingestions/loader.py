import os
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions

# <repo>/src/rag/data/policy_removed.pdf -- loader.py is at <repo>/src/rag/ingestions/
DEFAULT_PDF = Path(__file__).resolve().parents[1] / "data" / "policy_removed.pdf"


def recover_formula_text(document):
    """
    Fills in formula items from the text layer docling already extracted.

    Docling only writes `text` on a formula item when formula enrichment runs,
    so with enrichment off every equation serialises as
    `<!-- formula-not-decoded -->` and never reaches the index. The unicode the
    PDF's own text layer carried is still on the item as `orig`, so it is
    copied across when the enriched text is missing. Nothing is generated and
    nothing is guessed -- this only stops docling discarding what it parsed.

    Args:
        document: The docling document to repair, modified in place.

    Returns:
        The same document.
    """
    recovered = 0
    for item, _level in document.iterate_items():
        if "formula" not in str(getattr(item, "label", "")).lower():
            continue
        if (getattr(item, "text", "") or "").strip():
            continue  # enrichment produced a transcription; leave it alone
        original = (getattr(item, "orig", "") or "").strip()
        if original:
            item.text = original
            recovered += 1
    if recovered:
        print(f"Recovered {recovered} formula(s) from the PDF text layer.")
    return document


def loader(
    source=DEFAULT_PDF,
    do_ocr=False,
    do_table_structure=True,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
    generate_picture_images=False,
    num_threads=None,
):
    r"""
    Loads a PDF document and converts it into a structured format.

    The enrichment passes default to off, as they do in docling itself. Each
    one loads and runs an extra model over the page: formula enrichment in
    particular runs a generative vision model that decodes token by token, and
    on a CPU-only machine that turns a short PDF into a job of tens of minutes.
    They also degrade rather than fail there: formula enrichment emits runs of
    empty "\text { }" and picture description invents figure prose, both of
    which are then indexed as if they were document text. Equations survive
    without them via `recover_formula_text`, which reads the PDF's own text
    layer instead of running a model over the page image.

    Args:
        source: Path to the PDF to convert.
        do_ocr: Run OCR over the page images. Turn off for PDFs that already
            have a text layer -- it is the next biggest cost after enrichment.
        do_table_structure: Recover table rows and columns.
        do_formula_enrichment: Transcribe formulas with the CodeFormula model.
            Very slow without a GPU.
        do_picture_classification: Label figures with the figure classifier.
        do_picture_description: Describe pictures/figures with a multimodal model.
        generate_picture_images: Generate image assets for discovered pictures.
        num_threads: Torch threads to use; defaults to the CPU count.

    Returns:
        A docling document, or None if the conversion failed.
    """
    try:
        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            do_formula_enrichment=do_formula_enrichment,
            do_picture_classification=do_picture_classification,
            do_picture_description=do_picture_description,
            generate_picture_images=generate_picture_images,
            accelerator_options=AcceleratorOptions(
                num_threads=num_threads or os.cpu_count() or 4,
                device=AcceleratorDevice.AUTO,
            ),
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        print(f"Loading and converting {source}...")
        result = converter.convert(str(source))
        return recover_formula_text(result.document)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
