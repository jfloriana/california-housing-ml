from .pdf_generator import generate_pdf
from .word_generator import generate_word
from .excel_generator import generate_excel

def generate_report(format: str = "pdf", language: str = "es") -> bytes:
    if format == "pdf":
        return generate_pdf(language=language)
    elif format == "word":
        return generate_word(language=language)
    elif format == "excel":
        return generate_excel(language=language)
    else:
        raise ValueError(f"Unsupported format: {format}")
