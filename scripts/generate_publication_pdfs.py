#!/usr/bin/env python3
"""
Generate publication-ready PDFs from markdown documents.

Converts RESEARCH_ARTICLE_DRAFT.md and SUPPLEMENTAL_METHODS.md to PDFs
with proper LaTeX equation rendering and professional formatting.
"""

import subprocess
import sys
from pathlib import Path


def check_pandoc():
    """Check if pandoc is available."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"Found: {version}")
            return True
    except FileNotFoundError:
        pass
    return False


def convert_to_pdf(input_path: Path, output_path: Path, title: str = None):
    """
    Convert markdown to PDF using pandoc.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output PDF file
        title: Optional document title
    """
    cmd = [
        "pandoc",
        str(input_path),
        "-o", str(output_path),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "documentclass=article",
        "-V", "papersize=letter",
        "--highlight-style=tango",
        "--toc",
        "--toc-depth=3",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "-V", "urlcolor=blue",
    ]

    if title:
        cmd.extend(["-V", f"title={title}"])

    print(f"Converting {input_path.name} to PDF...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  SUCCESS: {output_path}")
        return True
    else:
        print(f"  ERROR: {result.stderr}")
        return False


def convert_with_python_markdown(input_path: Path, output_path: Path):
    """
    Fallback: Convert markdown to PDF using Python libraries.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output PDF file
    """
    try:
        import markdown
        from weasyprint import HTML, CSS

        # Read markdown
        with open(input_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML with math support
        html_content = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "codehilite", "md_in_html"]
        )

        # Wrap in HTML document with styling
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            max-width: 7.5in;
            margin: 1in auto;
            padding: 0 0.5in;
        }}
        h1 {{ font-size: 18pt; margin-top: 24pt; }}
        h2 {{ font-size: 14pt; margin-top: 18pt; }}
        h3 {{ font-size: 12pt; margin-top: 14pt; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12pt 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #f5f5f5; }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 12px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 3px solid #ccc;
            margin-left: 0;
            padding-left: 16px;
            color: #555;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

        # Convert to PDF
        HTML(string=full_html).write_pdf(str(output_path))
        print(f"  SUCCESS (weasyprint): {output_path}")
        return True

    except ImportError as e:
        print(f"  Missing library: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Main entry point."""
    # Paths
    docs_dir = Path(__file__).resolve().parents[1] / "_local" / "psc" / "reporting"
    output_dir = docs_dir / "pdfs"
    output_dir.mkdir(exist_ok=True)

    # Documents to convert
    documents = [
        {
            "input": docs_dir / "RESEARCH_ARTICLE_DRAFT.md",
            "output": output_dir / "Research_Article.pdf",
            "title": "Automated Detection of Research Front Inflections"
        },
        {
            "input": docs_dir / "SUPPLEMENTAL_METHODS.md",
            "output": output_dir / "Supplemental_Methods.pdf",
            "title": "Supplemental Methods"
        }
    ]

    print("=" * 60)
    print("Publication PDF Generation")
    print("=" * 60)

    # Check for pandoc
    has_pandoc = check_pandoc()

    success_count = 0
    for doc in documents:
        print(f"\nProcessing: {doc['input'].name}")

        if not doc["input"].exists():
            print(f"  SKIP: File not found")
            continue

        if has_pandoc:
            if convert_to_pdf(doc["input"], doc["output"], doc["title"]):
                success_count += 1
        else:
            print("  Pandoc not found, trying Python fallback...")
            if convert_with_python_markdown(doc["input"], doc["output"]):
                success_count += 1

    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{len(documents)} PDFs generated")

    if success_count < len(documents):
        print("\nTo generate PDFs with full LaTeX support, install:")
        print("  1. Pandoc: https://pandoc.org/installing.html")
        print("  2. LaTeX: MiKTeX (Windows) or TeX Live")
        print("\nOr install Python libraries:")
        print("  pip install markdown weasyprint")

    print("=" * 60)

    return 0 if success_count == len(documents) else 1


if __name__ == "__main__":
    sys.exit(main())
