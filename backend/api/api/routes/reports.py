import io
import importlib
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()

_REPORTS_MODULE = "backend.training.reports"


def _get_report_generator():
    try:
        mod = importlib.import_module(_REPORTS_MODULE)
        return mod
    except ModuleNotFoundError:
        raise HTTPException(
            status_code=501,
            detail=f"Reports module '{_REPORTS_MODULE}' is not available. "
                   "Install the training package or implement backend.training.reports",
        )


_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_FILE_EXTENSIONS = {
    "pdf": ".pdf",
    "word": ".docx",
    "excel": ".xlsx",
}


@router.get("/api/reports/generate")
async def generate_report(
    format: str = Query("pdf", regex="^(pdf|word|excel)$"),
    language: str = Query("es", regex="^(es|en)$"),
):
    if format not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    mod = _get_report_generator()

    try:
        if hasattr(mod, "generate_report"):
            result = mod.generate_report(format=format, language=language)
            if isinstance(result, bytes):
                content = result
            else:
                content = result.encode("utf-8") if isinstance(result, str) else result
        elif format == "pdf" and hasattr(mod, "generate_pdf"):
            content = mod.generate_pdf(language=language)
        elif format == "word" and hasattr(mod, "generate_word"):
            content = mod.generate_word(language=language)
        elif format == "excel" and hasattr(mod, "generate_excel"):
            content = mod.generate_excel(language=language)
        else:
            raise HTTPException(
                status_code=501,
                detail=f"Report generation for '{format}' is not implemented in {_REPORTS_MODULE}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}",
        )

    if not isinstance(content, bytes):
        if isinstance(content, str):
            content = content.encode("utf-8")
        else:
            content = str(content).encode("utf-8")

    filename = f"california_housing_report_{language}{_FILE_EXTENSIONS[format]}"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=_MEDIA_TYPES[format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        },
    )
