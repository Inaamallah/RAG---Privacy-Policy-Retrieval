import os
import re
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions

# <repo>/src/rag/data/policy_removed.pdf -- loader.py is at <repo>/src/rag/ingestions/
DEFAULT_PDF = Path(__file__).resolve().parents[1] / "data" / "policy_removed.pdf"


# A typeset equation is two-dimensional; a PDF text layer is not. A large
# delimiter is not one character -- the typesetter draws it as a stack of
# Unicode "piece" glyphs, an upper hook, any number of extensions, and a lower
# hook (U+239B..U+23AD for stretched parentheses, brackets and braces), and
# docling reads the stack out top to bottom, so one big bracket arrives as
# three separate characters. Each stack is folded back into the single bracket
# it was drawing: the extensions carry no shape of their own and are dropped,
# and the hooks are counted so that two delimiters standing next to each other
# do not collapse into one and leave the expression unbalanced.
_EXTENSION_GLYPHS = (
    "\u239c\u239f"      # parenthesis extensions
    "\u23a2\u23a5"      # bracket extensions
    "\u23a8\u23ac\u23aa"  # brace middles and extension
    "\ufe37\ufe38\u23de\u23df"  # the horizontal braces of an underbrace
)

# (upper hook, lower hook, the bracket the stack was drawing)
_DELIMITER_STACKS = (
    ("\u239b", "\u239d", "("),
    ("\u239e", "\u23a0", ")"),
    ("\u23a1", "\u23a3", "["),
    ("\u23a4", "\u23a6", "]"),
    ("\u23a7", "\u23a9", "{"),
    ("\u23ab", "\u23ad", "}"),
    ("\u23b0", "\u23b0", "{"),
    ("\u23b1", "\u23b1", "}"),
)


def normalize_formula_text(text):
    r"""
    Tidies the glyph sequence a PDF text layer gives for an equation.

    Folds every stack of stretched-delimiter pieces back into the bracket it
    drew, drops the extension glyphs and the horizontal braces of an underbrace
    annotation, and collapses the whitespace docling leaves between glyphs.
    Only typesetting debris is removed; no symbol is added, reordered, or
    reinterpreted.

    What this cannot repair is not repairable from the text layer at all: a
    fraction bar is a drawn rule that no character corresponds to, and scripts
    are expressed by baseline offset rather than markup. `log(t)/t` is
    extracted as `log(t) t`, and a subscripted variable as two tokens, whatever
    is done here -- see the note in `recover_formula_text`.

    Args:
        text: The raw text-layer string for one formula item.

    Returns:
        The tidied string.
    """
    text = text or ""
    for piece in _EXTENSION_GLYPHS:
        text = text.replace(piece, " ")

    for upper, lower, bracket in _DELIMITER_STACKS:
        pieces = f"{re.escape(upper)}{re.escape(lower)}"

        def fold(match, upper=upper, lower=lower, bracket=bracket):
            run = match.group(0)
            # A stack contributes one hook of each kind, so a run holding two
            # upper hooks is two delimiters -- not one drawn taller.
            return bracket * max(run.count(upper), run.count(lower), 1)

        text = re.sub(rf"(?:[{pieces}]\s*)+", fold, text)

    text = re.sub(r"\s+", " ", text)
    # Folding a stack leaves a gap inside the bracket it became.
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text.strip()


def recover_formula_text(document):
    """
    Fills in formula items from the text layer docling already extracted.

    Docling only writes `text` on a formula item when formula enrichment runs,
    so with enrichment off every equation serialises as
    `<!-- formula-not-decoded -->` and never reaches the index. The unicode the
    PDF's own text layer carried is still on the item as `orig`, so it is
    copied across -- through `normalize_formula_text` -- when the enriched text
    is missing. Nothing is generated and nothing is guessed; this only stops
    docling discarding what it parsed.

    **What arrives is a flattened equation, not a transcription.** The text
    layer of a PDF is a bag of positioned glyphs with no structure: a fraction
    bar is a drawn rule that no character corresponds to, and sub/superscripts
    are expressed by baseline offset rather than markup. Read out in order,
    `log(t)/t` becomes `log(t) t` and a subscripted variable becomes two
    tokens. `normalize_formula_text` removes the typesetting debris on top of
    that, but the lost relations cannot be recovered here, and downstream must
    not pretend otherwise -- hence rule 6 in `generation.prompts.SYSTEM_PROMPT`,
    which forbids the model reconstructing these strings into LaTeX. Run
    `rag ingest --formula-enrichment` when the equations themselves matter:
    that transcribes them with a model instead of reading the text layer.

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
        original = normalize_formula_text(getattr(item, "orig", ""))
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
