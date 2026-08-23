from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def loader():
    """
    Loads a PDF document and converts it into a structured format.
    """
    try:
        pipeline_options = PdfPipelineOptions(
                do_formula_enrichment = True,
                do_picture_classification = True,
                generate_picture_images = True
        )
        converter = DocumentConverter(format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options)})        

        print("Loading and converting the PDF document...")
        result = converter.convert(r"C:\Users\123\OneDrive - Higher Education Commission\Desktop\Study Material\Projects\RAG---Privacy-Policy-Retrieval\src\rag\data\policy.pdf")
        document = result.document
        return document

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


