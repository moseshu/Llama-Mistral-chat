from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd


def _score_dataframe(df: pd.DataFrame) -> float:
    """Heuristic score: prefer tables with more rows/cols and non-null density."""
    if df is None or df.empty:
        return 0.0
    rows, cols = df.shape
    non_null = float(df.notna().sum().sum())
    total = float(rows * max(cols, 1))
    density = (non_null / total) if total > 0 else 0.0
    # Favor at least 2 columns
    width_bonus = 1.0 if cols >= 2 else 0.2
    return rows * cols * density * width_bonus


def _pick_best(tables: Sequence[pd.DataFrame]) -> Optional[pd.DataFrame]:
    best_df: Optional[pd.DataFrame] = None
    best_score = 0.0
    for df in tables:
        score = _score_dataframe(df)
        if score > best_score:
            best_score = score
            best_df = df
    return best_df


def read_excel(path: Path) -> List[pd.DataFrame]:
    # Reads first sheet by default; supports multiple sheets
    xls = pd.ExcelFile(path)
    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        try:
            frames.append(pd.read_excel(xls, sheet_name=sheet))
        except Exception:
            continue
    return [f for f in frames if not f.empty]


def read_html_tables(path: Path) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(str(path))
        return [t for t in tables if not t.empty]
    except Exception:
        return []


def read_docx_tables(path: Path) -> List[pd.DataFrame]:
    try:
        from docx import Document  # type: ignore
    except Exception:
        return []
    try:
        doc = Document(str(path))
    except Exception:
        return []
    frames: List[pd.DataFrame] = []
    for tbl in doc.tables:
        rows = []
        for r in tbl.rows:
            rows.append([c.text.strip() for c in r.cells])
        if not rows:
            continue
        # First row as header if all cells non-empty
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        try:
            df = pd.DataFrame(body, columns=header)
        except Exception:
            df = pd.DataFrame(rows)
        frames.append(df)
    return [f for f in frames if not f.empty]


def read_pdf_tables(path: Path) -> List[pd.DataFrame]:
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return []
    frames: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for tbl in tables or []:
                    if not tbl:
                        continue
                    # Assume first row may be header
                    if len(tbl) > 1 and all(cell is not None for cell in tbl[0]):
                        header = [str(c).strip() for c in tbl[0]]
                        data = [[str(c).strip() for c in row] for row in tbl[1:]]
                        try:
                            df = pd.DataFrame(data, columns=header)
                        except Exception:
                            df = pd.DataFrame(tbl)
                    else:
                        df = pd.DataFrame(tbl)
                    frames.append(df)
    except Exception:
        return []
    # Clean whitespace-only columns
    cleaned: List[pd.DataFrame] = []
    for f in frames:
        if f.empty:
            continue
        f2 = f.copy()
        f2.columns = [str(c).strip() for c in f2.columns]
        cleaned.append(f2)
    return [f for f in cleaned if not f.empty]


def load_any_table(input_path: str) -> pd.DataFrame:
    """Load best-effort table from multiple file formats.

    Supported: CSV, TSV, JSON (array), Excel, HTML, DOCX, PDF.
    Returns the "best" table by heuristic scoring.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    suffix = path.suffix.lower()
    # Native support via pandas
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, engine="python")
    if suffix == ".json":
        return pd.read_json(path, orient="records", lines=False)

    candidates: List[pd.DataFrame] = []
    if suffix in {".xlsx", ".xls"}:
        candidates += read_excel(path)
    elif suffix in {".html", ".htm"}:
        candidates += read_html_tables(path)
    elif suffix in {".docx"}:
        candidates += read_docx_tables(path)
    elif suffix in {".pdf"}:
        candidates += read_pdf_tables(path)
    else:
        # Try generic attempts: HTML tables
        candidates += read_html_tables(path)
        # Optional OCR for images if dependencies available
        if suffix in {".png", ".jpg", ".jpeg"}:
            try:
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore
            except Exception:
                pytesseract = None  # type: ignore
                Image = None  # type: ignore
            if pytesseract and Image:
                try:
                    img = Image.open(str(path))
                    text = pytesseract.image_to_string(img)
                    # Very naive TSV-like split: lines then whitespace
                    rows = [
                        [cell for cell in line.split() if cell]
                        for line in text.splitlines()
                        if line.strip()
                    ]
                    if rows and max(len(r) for r in rows) >= 2:
                        max_len = max(len(r) for r in rows)
                        normalized = [r + [None] * (max_len - len(r)) for r in rows]
                        df = pd.DataFrame(normalized)
                        candidates.append(df)
                except Exception:
                    pass

    best = _pick_best(candidates)
    if best is None or best.empty:
        raise ValueError(f"Could not extract a tabular dataset from: {input_path}")
    return best

