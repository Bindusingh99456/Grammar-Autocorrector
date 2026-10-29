import html
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.engine import get_engine, correct_text


MAX_CHARACTERS = 2000


def build_inline_diff(original: str, edits: list) -> str:
    """
    Renders changed words inline: old struck through (<del class="diff-old">),
    new underlined (<ins class="diff-new">).
    """
    if not edits:
        return html.escape(original)

    # Sort edits by starting character offset
    sorted_edits = sorted(edits, key=lambda e: e.get("start", 0))

    parts = []
    cursor = 0
    for edit in sorted_edits:
        start = edit.get("start", 0)
        end = edit.get("end", 0)
        old_val = edit.get("old", "")
        new_val = edit.get("new", "")

        # Bounds check against the string length and cursor position
        if start < cursor or start > len(original) or end > len(original):
            continue

        # Append unmodified preceding text
        parts.append(html.escape(original[cursor:start]))

        # Append inline difference markers
        parts.append(
            f'<del class="diff-old">{html.escape(old_val)}</del> '
            f'<ins class="diff-new">{html.escape(new_val)}</ins>'
        )
        cursor = end

    # Append any remaining unmodified text
    parts.append(html.escape(original[cursor:]))
    return "".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the engine and model on startup
    print("Initializing Grammar Autocorrector engine...", flush=True)
    get_engine()
    yield


app = FastAPI(
    title="Grammar Autocorrector",
    description="Grammar Autocorrector",
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure app/static directory exists and mount it
static_dir = Path("app/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates directory
templates_dir = Path("app/templates")
templates_dir.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav_path = static_dir / "favicon.svg"
    return FileResponse(fav_path, media_type="image/svg+xml")


class CorrectRequest(BaseModel):
    text: str = Field(..., description="Input text to correct (maximum 2,000 characters)")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint providing engine status."""
    engine = get_engine()
    return {
        "status": "ok",
        "engine": engine.engine_name,
        "model_loaded": engine.model is not None,
    }


@app.post("/api/correct")
async def api_correct(payload: CorrectRequest):
    """
    JSON API for grammar correction.
    Accepts text up to 2,000 characters and returns structured correction results.
    """
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty.",
        )

    if len(text) > MAX_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input text exceeds the maximum limit of {MAX_CHARACTERS} characters (received {len(text):,} characters).",
        )

    result = correct_text(text)
    return result


# ---------------------------------------------------------------------------
# Frontend Web Routes (Server-rendered HTML with Jinja2)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main editor view."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "editor",
            "input_text": "",
            "result": None,
            "inline_diff_html": "",
            "error_message": None,
        },
    )


@app.post("/correct", response_class=HTMLResponse)
async def form_correct(request: Request, text: str = Form("")):
    """Form submission endpoint returning server-rendered correction."""
    if not text or not text.strip():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_page": "editor",
                "input_text": "",
                "result": None,
                "inline_diff_html": "",
                "error_message": "Please enter some text to correct.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if len(text) > MAX_CHARACTERS:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "active_page": "editor",
                "input_text": text[:200],
                "result": None,
                "inline_diff_html": "",
                "error_message": f"Input text exceeds the maximum limit of {MAX_CHARACTERS} characters (received {len(text):,} characters).",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    result = correct_text(text)
    inline_diff_html = build_inline_diff(result["original"], result["edits"])

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "editor",
            "input_text": text,
            "result": result,
            "inline_diff_html": inline_diff_html,
            "error_message": None,
        },
    )


@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    """How It Works page displaying pipeline and worked example."""
    return templates.TemplateResponse(
        request=request,
        name="how_it_works.html",
        context={
            "active_page": "how_it_works",
        },
    )


@app.get("/test-cases", response_class=HTMLResponse)
async def test_cases(request: Request):
    """Test cases table with Run It action buttons."""
    return templates.TemplateResponse(
        request=request,
        name="test_cases.html",
        context={
            "active_page": "test_cases",
        },
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page with project, author, and dataset information."""
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "active_page": "about",
        },
    )
