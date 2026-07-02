"""PDF report rendering utilities using fpdf2.

Provides reusable components for building structured, publication-quality
PDF reports: title pages, section headers, body text, tables, and embedded
figures. Designed for the final report and executive summary deliverables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

_FONT_DIRS = [
    Path("/usr/share/fonts/TTF"),
    Path("/usr/local/share/fonts/TTF"),
    Path(__file__).parent.parent.parent.parent / "fonts",
]


def _find_font(filename: str) -> str:
    """Locate a TrueType font file by searching known font directories.

    Searches ``_FONT_DIRS`` and falls back to matplotlib's bundled fonts.

    Args:
        filename: Font file name (e.g., ``"DejaVuSans.ttf"``).

    Returns:
        Absolute path to the font file.

    Raises:
        FileNotFoundError: If the font cannot be found in any search path.
    """
    for d in _FONT_DIRS:
        p = d / filename
        if p.exists():
            return str(p)
    # Fallback: matplotlib bundled fonts
    import matplotlib

    mpl_font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    if mpl_font.exists():
        return str(mpl_font)
    raise FileNotFoundError(
        f"Cannot find font file '{filename}'. " f"Searched: {[str(d) for d in _FONT_DIRS]}"
    )


class ReportPDF(FPDF):
    """Extended FPDF with report-specific styling and helpers.

    Attributes:
        section_numbering: Whether to auto-number sections.
        _section_counts: Running counter for section numbering.
    """

    def __init__(self, title: str, section_numbering: bool = True) -> None:
        """Initialize the PDF report with title and section numbering."""
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        self._title = title
        self.section_numbering = section_numbering
        self._section_counts: list[int] = []
        self._has_title_page = False

        # Add Unicode font (DejaVu) for wide glyph support
        font_regular = _find_font("DejaVuSans.ttf")
        font_bold = _find_font("DejaVuSans-Bold.ttf")
        self.add_font("DejaVu", "", font_regular)
        self.add_font("DejaVu", "B", font_bold)

    def _header(self) -> None:
        """Render the page header with title and page number.

        Displayed on every page after the title page (page > 1).
        """
        if self.page_no() > 1:
            self.set_fill_color(0, 0, 0)
            self.rect(0, 0, 210, 5, style="F")
            self.set_y(6)
            self.set_font("DejaVu", "B", 8)
            self.set_text_color(0, 0, 0)
            self.cell(0, 6, self._title, align="L")
            self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(0, 0, 0)
            self.line(10, self.get_y(), 200, self.get_y())
            self.set_y(self.get_y() + 2)

    def _footer(self) -> None:
        """Render the page footer with generation timestamp.

        Displayed on every page after the title page (page > 1).
        """
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("DejaVu", "", 7)
            self.set_text_color(160, 160, 160)
            gen_date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            self.cell(0, 10, f"Generated: {gen_date}", align="C")

    def add_title_page(self, subtitle: str, authors: str = "") -> None:
        """Render a professional title page with accent color."""
        self._has_title_page = True
        self.add_page()
        # Top accent bar
        self.set_fill_color(0, 0, 0)
        self.rect(0, 0, 210, 8, style="F")
        self.ln(55)
        # Title
        self.set_font("DejaVu", "B", 24)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 12, self._title, align="C")
        self.ln(6)
        # Horizontal rule
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.8)
        self.line(30, self.get_y(), 180, self.get_y())
        self.set_line_width(0.2)
        self.ln(8)
        # Subtitle
        self.set_font("DejaVu", "", 14)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 8, subtitle, align="C")
        if authors:
            self.ln(16)
            self.set_font("DejaVu", "", 11)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 7, authors, align="C")
        # Date and confidentiality — pinned 30mm from bottom
        self.set_y(240)
        self.set_font("DejaVu", "", 10)
        self.set_text_color(140, 140, 140)
        gen_date = datetime.now(UTC).strftime("%B %d, %Y")
        self.cell(0, 10, f"Generated: {gen_date}", align="C")
        self.ln(7)
        # Bottom accent bar — absolute position at very bottom
        self.set_fill_color(0, 0, 0)
        self.rect(0, 290, 210, 10, style="F")
        self.add_page()

    def add_section(self, title: str, level: int = 1) -> None:
        """Add a section heading at the given outline level.

        Args:
            title: Section title text.
            level: 1 for top-level sections, 2 for subsections, etc.
        """
        if self.section_numbering:
            while len(self._section_counts) < level:
                self._section_counts.append(0)
            self._section_counts = self._section_counts[:level]
            self._section_counts[level - 1] += 1
            numbering = ".".join(str(c) for c in self._section_counts)
            full_title = f"{numbering}  {title}"
        else:
            full_title = title

        sizes = {1: 16, 2: 13, 3: 11}
        weights = {1: "B", 2: "B", 3: ""}
        font_size = sizes.get(level, 11)
        font_weight = weights.get(level, "")

        self.ln(4)
        self.set_font("DejaVu", font_weight, font_size)
        self.set_text_color(30, 30, 30)
        if level == 1:
            # Black filled background band for top-level sections
            self.set_fill_color(0, 0, 0)
            self.set_text_color(255, 255, 255)
            self.set_font("DejaVu", "B", font_size)
            self.cell(0, 10, full_title, fill=True, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
            return
        elif level == 2:
            self.set_draw_color(0, 0, 0)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)
        self.multi_cell(0, 8, full_title)
        self.ln(2)

    def add_paragraph(self, text: str) -> None:
        """Add a body paragraph."""
        self.set_font("DejaVu", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def add_bullet(self, text: str, indent: int = 10) -> None:
        """Add a bullet point."""
        self.set_font("DejaVu", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(indent, 6, "•", new_x="END")
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_table(
        self, headers: list[str], rows: list[list[Any]], col_widths: list[float] | None = None
    ) -> None:
        """Add a table with header row and data rows.

        Args:
            headers: Column header strings.
            rows: List of row data (each row is a list of values).
            col_widths: Optional list of column widths in mm. Auto-calculated if None.
        """
        if col_widths is None:
            usable = 190
            col_widths = [usable / len(headers)] * len(headers)

        # Header row
        self.set_font("DejaVu", "B", 9)
        self.set_fill_color(0, 0, 0)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("DejaVu", "", 8)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
                # Re-draw header
                self.set_font("DejaVu", "B", 9)
                self.set_fill_color(0, 0, 0)
                self.set_text_color(255, 255, 255)
                for i, header in enumerate(headers):
                    self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
                self.ln()
                self.set_font("DejaVu", "", 8)
                self.set_text_color(0, 0, 0)

            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)

            for i, val in enumerate(row):
                text = str(val) if val is not None else ""
                self.cell(col_widths[i], 6, text, border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)

    def add_figure(self, image_path: str | Path, caption: str, width_mm: float = 160) -> None:
        """Embed a figure with caption.

        Args:
            image_path: Path to the image file.
            caption: Figure caption text.
            width_mm: Display width in mm (default 160).
        """
        path = Path(image_path)
        if not path.exists():
            self.add_paragraph(f"[Figure not found: {path.name}]")
            return

        if self.get_y() > 210:
            self.add_page()

        # Center the image
        x_centered = (210 - width_mm) / 2
        self.image(str(path), x=x_centered, w=width_mm)
        self.ln(2)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4.5, caption, align="C")
        self.ln(3)

    def add_insight_box(
        self,
        insight_id: str,
        observation: str,
        evidence: dict[str, Any],
        interpretation: str,
        recommendation: str,
        limitation: str,
        title: str = "",
    ) -> None:
        """Render a full business insight with the five-part structure.

        Args:
            insight_id: Insight identifier (e.g., INS-01).
            observation: Observation text.
            evidence: Statistical evidence dictionary.
            interpretation: Business interpretation.
            recommendation: Practical recommendation.
            limitation: Limitation text.
        """
        self.ln(2)
        # Insight header bar
        self.set_fill_color(220, 220, 220)
        self.set_draw_color(0, 0, 0)
        display_title = title if title else evidence.get("metric", "Analysis")
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, f"{insight_id}: {display_title}", fill=True)
        self.ln(3)

        # Observation
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, "Observation")
        self.ln(7)
        self.set_font("DejaVu", "", 9)
        self.multi_cell(0, 5, observation)
        self.ln(3)

        # Statistical evidence table
        self.set_font("DejaVu", "B", 10)
        self.cell(0, 6, "Statistical Evidence")
        self.ln(7)
        ev_rows = []
        for key, label in [
            ("test_name", "Test"),
            ("metric", "Metric"),
            ("groups", "Groups"),
            ("statistic", "Statistic"),
            ("p_value", "p-value"),
            ("effect_size", "Effect Size"),
            ("effect_size_measure", "Effect Size Measure"),
            ("corrected_threshold", "Corrected Threshold"),
            ("underpowered", "Underpowered"),
        ]:
            val = evidence.get(key, "")
            if key == "groups" and isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            ev_rows.append([label, str(val)])

        ev_col_widths = [50, 140]
        self.set_font("DejaVu", "", 8)
        self.set_text_color(50, 50, 50)
        fill = False
        for row in ev_rows:
            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            self.cell(ev_col_widths[0], 5.5, row[0], border=1, fill=True)
            self.cell(ev_col_widths[1], 5.5, row[1], border=1, fill=True)
            self.ln()
            fill = not fill
        self.ln(3)

        # Interpretation, Recommendation, Limitation
        for section_title, section_text in [
            ("Business Interpretation", interpretation),
            ("Practical Recommendation", recommendation),
            ("Limitation", limitation),
        ]:
            self.set_font("DejaVu", "B", 10)
            self.cell(0, 6, section_title)
            self.ln(7)
            self.set_font("DejaVu", "", 9)
            self.multi_cell(0, 5, section_text)
            self.ln(2)

        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
