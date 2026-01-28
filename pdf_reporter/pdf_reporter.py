"""
pdf_reporter.py

A lightweight PDF reporting helper built on ReportLab.
Provides a Report class with methods to add text, images, and tables, with sensible defaults and formatting.

"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from PyPDF2 import PdfMerger

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


# -----------------------------
# Formatting helpers
# -----------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _custom_round(x: Any, precision: int = 3) -> str:
    """
    Robust numeric formatting:
    - scientific notation for very large/small magnitude
    - fixed decimals otherwise
    """
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (pd.Timestamp, pd.Timedelta)):
            return str(x)
        if isinstance(x, (pd.NA.__class__,)):
            return ""
        if _is_number(x):
            if math.isnan(x):
                return "NaN"
            if math.isinf(x):
                return "Inf" if x > 0 else "-Inf"
            ax = abs(x)
            if ax > 1e5 or (0 < ax < 1e-5):
                return f"{x:.{precision}e}"
            return f"{x:.{precision}f}"
        return str(x)
    except Exception:
        return str(x)

def _escape_html(s: str) -> str:
    # ReportLab Paragraph uses a mini-HTML parser; escape the important characters.
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

def _ellipsize(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    if max_chars <= 3:
        return s[:max_chars]
    return s[: max_chars - 3] + "..."

def _fit_title_font_size(
    text: str,
    max_width_pt: float,
    start_size: int = 28,
    min_size: int = 14,
    font_name: str = "Helvetica-Bold",
) -> int:
    """
    Reduce font size until the title fits on one line OR we hit min_size.
    Paragraph will still wrap if needed, but this prevents comically oversized titles.
    """
    text = text or ""
    size = start_size
    # Use stringWidth on a sanitized string (no markup).
    plain = text.replace("\n", " ").strip()
    while size > min_size and stringWidth(plain, font_name, size) > max_width_pt:
        size -= 1
    return size


# -----------------------------
# Canvas with page numbering
# -----------------------------

class NumberedCanvas(Canvas):
    def showPage(self):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        x = self._pagesize[0] / 2.0
        self.drawCentredString(x, 0.35 * inch, str(self._pageNumber))
        Canvas.showPage(self)


# -----------------------------
# Report class (public API)
# -----------------------------

class _TablePreviewConfig:
    def __init__(
        self,
        max_rows: int = 16,
        head_rows: int = 12,
        tail_rows: int = 3,
        max_cols: int = 14,
        head_cols: int = 6,
        tail_cols: int = 5,
        ellipsis_token: str = '…',
    ):
        self.max_rows = max_rows
        self.head_rows = head_rows
        self.tail_rows = tail_rows
        self.max_cols = max_cols
        self.head_cols = head_cols
        self.tail_cols = tail_cols
        self.ellipsis_token = ellipsis_token

class Report:
    """
    Backward-compatible PDF report builder.

    IMPORTANT: Keep method signatures stable to avoid breaking PySPRESSO.
    """

    def __init__(
        self,
        name: str,
        title: str = "",
        elements: Optional[list] = None,
        rightMargin: int = 20,
        leftMargin: int = 20,
        topMargin: int = 20,
        bottomMargin: int = 20,
    ):
        self.title = title
        if not name.endswith(".pdf"):
            name = name + ".pdf"

        self.elements = [] if elements is None else elements

        self.doc = SimpleDocTemplate(
            name,
            pagesize=letter,
            rightMargin=rightMargin,
            leftMargin=leftMargin,
            topMargin=topMargin,
            bottomMargin=bottomMargin,
        )

        page_w, page_h = letter
        self.page_width = page_w - rightMargin - leftMargin
        self.page_height = page_h - topMargin - bottomMargin

        self.name = name

        # Styles
        base = getSampleStyleSheet()

        self.styles = {
            "title": base["Title"],
            "normal": base["Normal"],
            "italic": base["Italic"],
        }

        self.styles["cover_title"] = ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f4e79"),
            spaceAfter=10,
        )
        self.styles["cover_subtitle"] = ParagraphStyle(
            "cover_subtitle",
            parent=base["Normal"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f4e79"),
            fontSize=12,
            leading=14,
            spaceAfter=10,
        )
        self.styles["cover_note"] = ParagraphStyle(
            "cover_note",
            parent=base["Italic"],
            alignment=TA_CENTER,
            fontSize=9,
            leading=11,
            textColor=colors.grey,
            spaceAfter=8,
        )

        self.styles["section"] = ParagraphStyle(
            "section",
            parent=base["Heading2"],
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=8,
            spaceAfter=6,
        )
        self.styles["caption"] = ParagraphStyle(
            "caption",
            parent=base["Italic"],
            alignment=TA_LEFT,
            fontSize=8.5,
            leading=10,
            textColor=colors.grey,
            spaceAfter=4,
        )

        self.styles["table_header"] = ParagraphStyle(
            "table_header",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
        self.styles["table_cell"] = ParagraphStyle(
            "table_cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=colors.black,
            alignment=TA_LEFT,
        )
        self.styles["table_cell_right"] = ParagraphStyle(
            "table_cell_right",
            parent=self.styles["table_cell"],
            alignment=TA_RIGHT,
        )

        # Table preview behavior (internal)
        self._table_preview = _TablePreviewConfig()

    # ---------------------------------------------
    # Cover / finalization
    # ---------------------------------------------

    def initialize_report(self):  
        """
        Initialize a nicer cover page.

        """
        title_text = self.title.strip() if isinstance(self.title, str) and self.title.strip() else os.path.splitext(os.path.basename(self.name))[0]
        title_text = _escape_html(title_text)

        # Fit the main title a bit better.
        fitted_size = _fit_title_font_size(title_text, max_width_pt=self.page_width * 0.92, start_size=30, min_size=16)
        cover_title_style = self.styles["cover_title"].clone("cover_title_fit")
        cover_title_style.fontName = "Helvetica-Bold"
        cover_title_style.fontSize = fitted_size
        cover_title_style.leading = fitted_size * 1.15

        self.elements.append(Spacer(1, 18))
        self.elements.append(Paragraph(title_text, cover_title_style))

    
        descriptor = "REPORT"
        self.elements.append(Paragraph(descriptor, self.styles["cover_subtitle"]))

        # A short intro (kept generic; references can be added elsewhere)
        intro = (
            "This document summarizes the dataset and the processing / analysis steps it went through. "
            "Figures are saved alongside the report (PNG and PDF) under their respective section names."
        )
        self.elements.append(Spacer(1, 6))
        self.elements.append(Paragraph(_escape_html(intro), self.styles["cover_note"]))

        # Divider
        self.elements.append(Spacer(1, 10))
        self.add_line()

        self.elements.append(Spacer(1, 8))

    def finalize_report(self):
        self.doc.build(self.elements, canvasmaker=NumberedCanvas)

    # ---------------------------------------------
    # Basic building blocks
    # ---------------------------------------------

    def add_pagebreak(self, return_element_only=False):
        if return_element_only:
            return PageBreak()
        self.elements.append(PageBreak())

    def add_line(self, return_element_only=False):
        # Use a thin subtle divider (
        line = LongTable([[""]], colWidths=[self.page_width])
        line.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (-1, -1), 0.8, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        if return_element_only:
            return line
        self.elements.append(line)

    def add_text(self, text, style="normal", alignment="left", font_size=10, return_element_only=False):
        """
        Add a text paragraph with specified style and alignment.
        """
    
        alignment_dict = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}

        if text is None:
            text = ""
        text = str(text)

        raw = str(text)
        # preserve existing <br/> tags
        raw = raw.replace("<br/>", "__BR__")
        raw = _escape_html(raw).replace("__BR__", "<br/>")
        if not raw.lstrip().startswith("<br/>"):
            raw = "<br/>" + raw
        text = raw

        if style == "bold":
            pstyle = self.styles["normal"].clone("bold_clone")
            pstyle.fontName = "Helvetica-Bold"
        elif style == "section":
            pstyle = self.styles["section"].clone("section_clone")
        else:
            pstyle = self.styles.get(style, self.styles["normal"]).clone("clone")

        pstyle.alignment = alignment_dict.get(alignment, TA_LEFT)
        pstyle.fontSize = font_size if style not in ("section",) else pstyle.fontSize
        pstyle.leading = max(pstyle.leading, pstyle.fontSize * 1.2)

        para = Paragraph(text, pstyle)

        if return_element_only:
            return [para, Spacer(1, 3)]
        self.elements.append(para)
        self.elements.append(Spacer(1, 3))

    # ---------------------------------------------
    # Index grid
    # ---------------------------------------------

    def _suggest_index_grid_cols(self, idxs: List[int]) -> int:
        """Pick a reasonable number of columns for an index grid based on page width and digit count."""
        if not idxs:
            return 12
        max_digits = max(len(str(abs(int(x)))) for x in idxs)
        # Rough per-cell minimum width in points: digits * ~4.5pt + padding
        min_cell_w = max(18.0, (max_digits * 4.6) + 12.0)
        cols = int(self.page_width // min_cell_w)
        return max(8, min(24, cols))

    def _build_index_grid_table(self, idxs: List[int], cols: int = 12) -> LongTable:
        """Render a compact, headerless grid of integers that can flow across pages."""
        if not idxs:
            t = LongTable([[""]], colWidths=[self.page_width])
            return t

        cols = max(1, int(cols))
        rows: List[List[str]] = []
        for i in range(0, len(idxs), cols):
            chunk = idxs[i : i + cols]
            row = [str(x) for x in chunk]
            if len(row) < cols:
                row += [""] * (cols - len(row))
            rows.append(row)

        # Equal widths work best for index grids
        col_w = self.page_width / cols
        table = LongTable(rows, colWidths=[col_w] * cols, repeatRows=0)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )

        # Subtle zebra for readability
        for r in range(len(rows)):
            if r % 2 == 1:
                table.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), colors.whitesmoke)]))

        return table

    def add_image(self, image, max_width=None, max_height=None, return_element_only=False):
        """
        Add an image (path or Image object), scaled to fit the usable page area.
        """
        if isinstance(image, str):
            if not (image.lower().endswith(".png") or image.lower().endswith(".jpg") or image.lower().endswith(".jpeg")):
                # keep legacy default
                image = image + ".png"
            if not os.path.exists(image):
                print(f"Warning: File not found - {image}")
            image = Image(image)

        # Scale to fit within the page content box, leaving some breathing room
        if max_width is not None:
            max_w = min(self.page_width, float(max_width))
        else:
            max_w = self.page_width
        if max_height is not None:
            max_h = min(self.page_height, float(max_height))
        else:
            max_h = self.page_height * 0.82

        width_scale = max_w / float(image.drawWidth)
        height_scale = max_h / float(image.drawHeight)
        scale = min(width_scale, height_scale, 1.0)

        image.drawWidth *= scale
        image.drawHeight *= scale
        image.hAlign = "CENTER"

        if return_element_only:
            return [image, Spacer(1, 8)]
        self.elements.append(image)
        self.elements.append(Spacer(1, 8))

    # ---------------------------------------------
    # Tables 
    # ---------------------------------------------

    def _make_table_preview(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Return a preview DataFrame (possibly reduced with ellipsis) + a human caption.

        Strategy:
        - rows: head + ellipsis + tail when too tall
        - cols: keep identifier-like columns + sample numeric columns across width when too wide
        """
        cfg = self._table_preview
        total_rows, total_cols = df.shape

        # -----------------
        # Row preview
        # -----------------
        if total_rows <= cfg.max_rows:
            df_rows = df.copy()
            row_note = f"{total_rows} rows"
        else:
            head = df.head(cfg.head_rows)
            tail = df.tail(cfg.tail_rows)
            ell = pd.DataFrame([[cfg.ellipsis_token] * total_cols], columns=df.columns)
            df_rows = pd.concat([head, ell, tail], ignore_index=True)
            row_note = f"{total_rows} rows (showing {len(df_rows)}: head {cfg.head_rows} + tail {cfg.tail_rows})"

        # -----------------
        # Column preview
        # -----------------
        if total_cols <= cfg.max_cols:
            df_view = df_rows.copy()
            col_note = f"{total_cols} columns"
            caption = f"Table shape: {total_rows} × {total_cols}. Preview shows {row_note} and {col_note}."
            return df_view, caption

        cols = list(df.columns)

        # Identifier-like columns to keep (in addition to the very first column)
        id_patterns = [
            r"id",
            r"cpd",
            r"compound",
            r"feature",
            r"m/?z",
            r"mz",
            r"rt",
            r"retention",
            r"annotation",
            r"class",
            r"group",
            r"sample",
        ]

        def is_id_col(name: str) -> bool:
            n = str(name).lower()
            return any(re.search(p, n) for p in id_patterns)

        keep: List[str] = []
        if cols:
            keep.append(cols[0])  # always keep the first column (often an ID)

        # Add up to 2 more "identifier-like" columns
        for c in cols[1:]:
            if c in keep:
                continue
            if is_id_col(c):
                keep.append(c)
            if len(keep) >= 3:
                break

        # We reserve 1 column slot for the ellipsis marker column.
        available = max(1, cfg.max_cols - 1)
        # Ensure keep does not exceed available.
        keep = keep[: min(len(keep), available)]

        remaining = [c for c in cols if c not in keep]
        slots_for_sample = max(0, available - len(keep))

        sampled: List[str] = []
        if remaining and slots_for_sample > 0:
            # Evenly sample across the remaining columns
            if slots_for_sample == 1:
                sampled = [remaining[len(remaining) // 2]]
            else:
                # Create target indices spaced across the range
                raw_idx = [int(round(i * (len(remaining) - 1) / (slots_for_sample - 1))) for i in range(slots_for_sample)]
                # De-duplicate while preserving order
                seen = set()
                idx = []
                for k in raw_idx:
                    if k not in seen:
                        seen.add(k)
                        idx.append(k)
                sampled = [remaining[k] for k in idx]
                # If duplicates reduced our count, fill with next unused columns
                if len(sampled) < slots_for_sample:
                    for c in remaining:
                        if c not in sampled:
                            sampled.append(c)
                        if len(sampled) >= slots_for_sample:
                            break

        chosen = (keep + sampled)[:available]

        df_view = df_rows[chosen].copy()
        # Insert an ellipsis column between keep and sampled if something was omitted
        omitted = total_cols - len(chosen)
        if omitted > 0:
            insert_at = len(keep)
            df_view.insert(insert_at, cfg.ellipsis_token, [cfg.ellipsis_token] * len(df_view))
            col_note = f"{total_cols} columns (showing {len(chosen)} + ellipsis; sampled across width, kept {len(keep)} ID/meta column(s))"
        else:
            col_note = f"{total_cols} columns (showing {len(chosen)})"

        caption = f"Table shape: {total_rows} × {total_cols}. Preview shows {row_note} and {col_note}."
        return df_view, caption

    def _compute_col_widths(self, rows: List[List[str]], max_total_width: float, font_size: float) -> List[float]:
        """
        Estimate column widths based on max string length per column and fit to page width.
        """
        if not rows:
            return [max_total_width]

        ncols = len(rows[0])

        # approximate average character width in points for Helvetica at given size
        # (0.52 is a decent heuristic; plus we clamp)
        char_w = max(3.2, font_size * 0.52)

        # measure max length per column (cap to keep sane)
        max_lens = [0] * ncols
        for r in rows[: min(len(rows), 200)]:  # limit scanning for huge previews
            for j, cell in enumerate(r):
                s = cell if cell is not None else ""
                s = str(s)
                max_lens[j] = max(max_lens[j], min(len(s), 60))

        base = []
        for L in max_lens:
            w = (L * char_w) + 10  # padding
            # Clamp to reasonable bounds in points
            w = max(28, min(140, w))
            base.append(w)

        total = sum(base)
        if total <= max_total_width:
            return base

        # Scale down proportionally, but keep a minimum width.
        min_w = 24
        scale = max_total_width / total
        widths = [max(min_w, w * scale) for w in base]

        # If still too wide due to mins, squeeze again.
        total2 = sum(widths)
        if total2 > max_total_width:
            over = total2 - max_total_width
            # reduce only columns above min_w
            adjustable = [i for i, w in enumerate(widths) if w > min_w + 1]
            if adjustable:
                reduce_each = over / len(adjustable)
                for i in adjustable:
                    widths[i] = max(min_w, widths[i] - reduce_each)

        return widths

    def add_table(self, data, return_element_only=False):
        """
        Add a DataFrame (recommended) as a nicely formatted, page-splitting table.

        Backward compatible: accepts pandas DataFrame. If a list-of-lists is passed,
        it will be treated as already tabular (first row assumed header if it's all strings).
        """
        if data is None:
            df = pd.DataFrame([[]])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            # best-effort conversion
            try:
                df = pd.DataFrame(data)
            except Exception:
                df = pd.DataFrame([[str(data)]])

        # Format values
        df = df.copy()

        # Ensure columns are strings
        df.columns = [str(c) for c in df.columns]

        # Table preview (avoid exploding PDFs)
        df_view, caption = self._make_table_preview(df)

        # Build text matrix, applying numeric formatting
        display = df_view.copy()
        for col in display.columns:
            display[col] = display[col].map(lambda x: _custom_round(x, precision=3))

        # Convert to string and ellipsize very long cells (wrapping still occurs)
        max_cell_chars = 120
        for col in display.columns:
            display[col] = display[col].map(lambda x: _ellipsize(_escape_html(str(x)), max_cell_chars))

        header = [_escape_html(str(c)) for c in display.columns.tolist()]
        body = display.values.tolist()
        body = [[str(x) for x in row] for row in body]

        # Determine per-column alignment (numeric => right)
        numeric_cols = []
        for col in display.columns:
            try:
                # treat as numeric if all non-empty entries parse as numbers
                series = df_view[col]
                ok = True
                for v in series:
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    if isinstance(v, str) and v.strip() in ("", self._table_preview.ellipsis_token):
                        continue
                    if not _is_number(v):
                        ok = False
                        break
                numeric_cols.append(ok)
            except Exception:
                numeric_cols.append(False)

        # Convert cells to Paragraphs (wrapping)
        table_rows: List[List[Any]] = []
        table_rows.append([Paragraph(h, self.styles["table_header"]) for h in header])

        for r in body:
            row_cells = []
            for j, cell in enumerate(r):
                st = self.styles["table_cell_right"] if numeric_cols[j] else self.styles["table_cell"]
                row_cells.append(Paragraph(cell, st))
            table_rows.append(row_cells)

        # Column widths fitted to page width
        width_estimation_rows = [header] + body
        col_widths = self._compute_col_widths(width_estimation_rows, max_total_width=self.page_width, font_size=self.styles["table_cell"].fontSize)

        # Caption with original shape and preview details
        caption_el = Paragraph(_escape_html(caption), self.styles["caption"])

        # Use LongTable for page splitting + repeat header
        table = LongTable(table_rows, colWidths=col_widths, repeatRows=1)

        # Styling
        header_bg = colors.HexColor("#1f4e79")
        zebra = colors.HexColor("#f3f6fa")

        ts = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("LINEBELOW", (0, 0), (-1, 0), 1.0, header_bg),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )

        # Zebra striping for body rows
        for i in range(1, len(table_rows)):
            if i % 2 == 0:
                ts.add("BACKGROUND", (0, i), (-1, i), zebra)

        # Emphasize first column (often an identifier) to help scanning wide numeric tables
        ts.add("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eef2f6"))
        ts.add("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold")

        # Column-level alignment tweaks: set whole column right if numeric
        for j, is_num in enumerate(numeric_cols):
            if is_num:
                ts.add("ALIGN", (j, 1), (j, -1), "RIGHT")
            else:
                ts.add("ALIGN", (j, 1), (j, -1), "LEFT")
        ts.add("ALIGN", (0, 0), (-1, 0), "CENTER")

        table.setStyle(ts)

        out_elements = [caption_el, table, Spacer(1, 8)]

        if return_element_only:
            return out_elements
        self.elements.extend(out_elements)

    # ---------------------------------------------
    # Grouping / merging (compatibility)
    # ---------------------------------------------

    def add_together(self, elements):  
        keep_together_elements = []
        for element in elements:
            if isinstance(element, tuple):
                type = element[0]
                if type == "text":
                    element = self.add_text(*element[1:], return_element_only=True)
                elif type == "image":
                    element = self.add_image(*element[1:], return_element_only=True)
                elif type == "table":
                    element = self.add_table(*element[1:], return_element_only=True)
                else:
                    raise ValueError("Invalid type: " + type)

                if isinstance(element, list):
                    keep_together_elements.extend(element)
                else:
                    keep_together_elements.append(element)

            elif isinstance(element, str):
                if element == "line":
                    keep_together_elements.append(self.add_line(return_element_only=True))
                elif element == "pagebreak":
                    keep_together_elements.append(self.add_pagebreak(return_element_only=True))
                else:
                    raise ValueError("Invalid type: " + element)
            else:
                raise ValueError("Invalid element: " + str(element))
        self.elements.append(KeepTogether(keep_together_elements))

    def merge_pdfs(self, pdfs, output_name):
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(output_name)
        merger.close()
        return output_name
