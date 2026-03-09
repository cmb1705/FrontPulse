#!/usr/bin/env python3
"""Convert markdown documents to publication-ready PDFs.

This script converts markdown files from the 2YP reporting directory to
professionally formatted PDFs suitable for academic publication.

Usage:
    python scripts/convert_to_pdf.py
"""

import re
from pathlib import Path
from fpdf import FPDF
import markdown


class AcademicPDF(FPDF):
    """Custom PDF class for academic papers with HTML support."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(True, margin=25)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', '', 10)
        self.cell(0, 10, str(self.page_no()), align='C')


def clean_text_for_pdf(text):
    """Clean text for PDF rendering."""
    # Unicode replacements for characters fpdf can't handle
    replacements = {
        '\u2192': '->',
        '\u2265': '>=',
        '\u2264': '<=',
        '\u00b1': '+/-',
        '\u0394': 'Delta',
        '\u03b3': 'gamma',
        '\u03bc': 'mu',
        '\u03c3': 'sigma',
        '\u2013': '-',
        '\u2014': '--',
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2022': '*',
        '\u2212': '-',
        '\u00d7': 'x',
        '\u2248': '~',
        '\u221e': 'inf',
        '\u2211': 'Sum',
        '\u220f': 'Product',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_markdown(text):
    """Clean markdown for better conversion."""
    # Remove LaTeX math (fpdf2 doesn't support it)
    text = re.sub(r'\$\$.*?\$\$', '[equation]', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]+\$', '[math]', text)

    return text


def convert_markdown_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Convert a markdown file to PDF.

    Args:
        md_path: Path to input markdown file
        pdf_path: Path to output PDF file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read markdown
        md_content = md_path.read_text(encoding='utf-8')

        # Clean markdown
        md_content = clean_markdown(md_content)
        md_content = clean_text_for_pdf(md_content)

        # Convert to HTML
        md_converter = markdown.Markdown(
            extensions=['tables', 'fenced_code', 'sane_lists']
        )
        html_content = md_converter.convert(md_content)

        # Wrap in basic HTML with styling
        styled_html = f"""
        <style>
            h1 {{ font-size: 16pt; font-weight: bold; }}
            h2 {{ font-size: 13pt; font-weight: bold; }}
            h3 {{ font-size: 11pt; font-weight: bold; }}
            p {{ font-size: 11pt; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #333; padding: 4px; }}
            th {{ background-color: #f0f0f0; }}
            code {{ font-family: Courier; font-size: 9pt; }}
        </style>
        {html_content}
        """

        # Create PDF
        pdf = AcademicPDF()
        pdf.set_margins(25, 25, 25)
        pdf.add_page()
        pdf.set_font('Times', '', 11)

        # Use write_html which handles layout automatically
        pdf.write_html(styled_html)

        # Save PDF
        pdf.output(str(pdf_path))

        print(f"Created: {pdf_path}")
        print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        print(f"  Pages: {pdf.page_no()}")
        return True

    except Exception as e:
        print(f"Error converting {md_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function."""
    base_dir = Path(__file__).resolve().parents[1] / "_local" / "psc" / "reporting"

    print("=" * 60)
    print("PDF CONVERSION FOR PUBLICATION")
    print("=" * 60)
    print()

    # Track success
    success_count = 0

    # Convert Research Article
    print("Converting Research Article...")
    if convert_markdown_to_pdf(
        base_dir / "RESEARCH_ARTICLE_DRAFT.md",
        base_dir / "Research_Article.pdf"
    ):
        success_count += 1
    print()

    # Convert Supplemental Methods
    print("Converting Supplemental Methods...")
    if convert_markdown_to_pdf(
        base_dir / "SUPPLEMENTAL_METHODS.md",
        base_dir / "Supplemental_Methods.pdf"
    ):
        success_count += 1
    print()

    print("=" * 60)
    print(f"CONVERSION COMPLETE: {success_count}/2 documents processed")
    print("=" * 60)

    if success_count == 2:
        print("\nOutput files:")
        print(f"  1. {base_dir / 'Research_Article.pdf'}")
        print(f"  2. {base_dir / 'Supplemental_Methods.pdf'}")
        return 0
    else:
        print("\nSome conversions failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
