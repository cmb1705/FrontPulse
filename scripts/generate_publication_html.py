#!/usr/bin/env python3
"""
Generate publication-ready HTML from markdown documents with embedded figures.
"""

import base64
import re
from collections.abc import Iterable
from pathlib import Path


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 data URI."""
    if not image_path.exists():
        return ""

    suffix = image_path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    }
    mime_type = mime_types.get(suffix, 'image/png')

    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{data}"


def protect_latex(text: str) -> tuple[str, dict]:
    """Extract LaTeX expressions and replace with placeholders to protect from markdown processing."""
    placeholders = {}
    counter = [0]  # Use list to allow modification in nested function

    def replace_latex(match):
        placeholder = f"__LATEX_PLACEHOLDER_{counter[0]}__"
        placeholders[placeholder] = match.group(0)
        counter[0] += 1
        return placeholder

    # Protect display-style math first, then inline math. Use non-greedy matching.
    text = re.sub(
        r'\\begin\{([a-zA-Z*]+)\}(.+?)\\end\{\1\}',
        replace_latex,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r'\\\[(.+?)\\\]', replace_latex, text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.+?)\$\$', replace_latex, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', replace_latex, text, flags=re.DOTALL)
    text = re.sub(r'\$([^$\n]+?)\$', replace_latex, text)
    return text, placeholders


def restore_latex(text: str, placeholders: dict) -> str:
    """Restore LaTeX expressions from placeholders."""
    for placeholder, latex in placeholders.items():
        text = text.replace(placeholder, latex)
    return text


def markdown_to_html(
    md_content: str,
    base_path: Path,
    display_title_html: str | None = None,
    page_break_before_headings: Iterable[str] | None = None,
) -> str:
    """Convert markdown to HTML with proper image embedding and LaTeX preservation."""
    # Protect LaTeX expressions before processing
    md_content, latex_placeholders = protect_latex(md_content)
    page_break_before = set(page_break_before_headings or [])

    lines = md_content.split("\n")
    html_lines = []
    in_code_block = False
    in_indented_code = False
    in_table = False
    table_rows = []
    in_ul = False
    in_ol = False
    used_display_title = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    def is_display_math_placeholder(line: str) -> bool:
        match = re.fullmatch(r"\s*(__LATEX_PLACEHOLDER_\d+__)\s*", line)
        if not match:
            return False
        latex = latex_placeholders.get(match.group(1), "").strip()
        return (
            latex.startswith("$$")
            or latex.startswith(r"\[")
            or latex.startswith(r"\begin")
        )

    def render_heading(level: int, text: str) -> None:
        nonlocal used_display_title
        close_lists()
        if text in page_break_before:
            html_lines.append('<div class="page-break"></div>')
        if level == 1 and display_title_html and not used_display_title:
            text = display_title_html
            used_display_title = True
        html_lines.append(f"<h{level}>{text}</h{level}>")

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            close_lists()
            if in_indented_code:
                html_lines.append("</code></pre>")
                in_indented_code = False
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                lang = line[3:].strip() or "text"
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Indented code blocks
        if in_indented_code:
            if line.startswith("    "):
                html_lines.append(line[4:].replace("<", "&lt;").replace(">", "&gt;"))
                continue
            if not line.strip():
                html_lines.append("")
                continue
            html_lines.append("</code></pre>")
            in_indented_code = False

        if line.startswith("    ") and line.strip():
            close_lists()
            html_lines.append('<pre><code class="language-text">')
            html_lines.append(line[4:].replace("<", "&lt;").replace(">", "&gt;"))
            in_indented_code = True
            continue

        # Images - convert to embedded base64
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            close_lists()
            alt_text = img_match.group(1)
            img_src = img_match.group(2)

            # Resolve image path relative to base_path
            img_path = base_path / img_src
            if img_path.exists():
                data_uri = image_to_base64(img_path)
                html_lines.append('<figure>')
                html_lines.append(f'<img src="{data_uri}" alt="{alt_text}" style="max-width:100%; height:auto;">')
                html_lines.append('</figure>')
            else:
                html_lines.append(f'<p><em>[Image not found: {img_src}]</em></p>')
            continue

        # Figure captions (italicized text starting with "Figure")
        if line.startswith('*Figure') and line.endswith('*'):
            close_lists()
            caption = line[1:-1]  # Remove asterisks
            html_lines.append(f'<figcaption><em>{caption}</em></figcaption>')
            continue

        # Tables - require line to start AND end with | (proper markdown table format)
        # This prevents matching things like |t - t₀| (absolute value) or formulas with |
        is_table_line = (line.strip().startswith("|") and line.strip().endswith("|")
                         and line.count("|") >= 3 and not line.startswith("    "))
        if is_table_line:
            close_lists()
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue  # Skip separator row
            if not in_table:
                # Center all tables for consistent, clean appearance
                html_lines.append('<table class="centered-table">')
                in_table = True
            tag = "th" if not table_rows else "td"
            # Process bold/italic/code in table cells
            processed_cells = []
            for c in cells:
                c = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", c)
                c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", c)
                c = re.sub(r"\*(.+?)\*", r"<em>\1</em>", c)
                c = re.sub(r"`([^`]+)`", r"<code>\1</code>", c)
                processed_cells.append(c)
            row = "".join(f"<{tag}>{c}</{tag}>" for c in processed_cells)
            html_lines.append(f"<tr>{row}</tr>")
            table_rows.append(cells)
            continue
        elif in_table:
            html_lines.append("</table>")
            in_table = False
            table_rows = []

        # Headers
        if line.startswith("######"):
            render_heading(6, line[6:].strip())
        elif line.startswith("#####"):
            render_heading(5, line[5:].strip())
        elif line.startswith("####"):
            render_heading(4, line[4:].strip())
        elif line.startswith("###"):
            render_heading(3, line[3:].strip())
        elif line.startswith("##"):
            render_heading(2, line[2:].strip())
        elif line.startswith("#"):
            render_heading(1, line[1:].strip())
        # Horizontal rules
        elif line.strip() in ("---", "***", "___"):
            close_lists()
            html_lines.append("<hr>")
        # Blockquotes
        elif line.startswith(">"):
            close_lists()
            html_lines.append(f"<blockquote>{line[1:].strip()}</blockquote>")
        # Standalone display math
        elif is_display_math_placeholder(line):
            close_lists()
            html_lines.append(f'<div class="display-math">{line.strip()}</div>')
        # Lists
        elif line.strip().startswith("- "):
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            content = line.strip()[2:]
            # Process code formatting in list items
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        elif re.match(r"^\d+\.\s", line.strip()):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            content = re.sub(r"^\d+\.\s", "", line.strip())
            # Process code formatting in list items
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        # Paragraphs
        elif line.strip():
            close_lists()
            # Bold and italic
            processed = line
            processed = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", processed)
            processed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", processed)
            processed = re.sub(r"\*(.+?)\*", r"<em>\1</em>", processed)
            processed = re.sub(r"`([^`]+)`", r"<code>\1</code>", processed)
            # Em dash
            processed = processed.replace("—", "&mdash;")
            # Table notes (starting with *Note: or similar) should clear floats
            if line.strip().startswith("*Note") or line.strip().startswith("*Total"):
                html_lines.append(f'<p class="table-note">{processed}</p>')
            # Bold section headings (like **Retrospective Evaluation:**) should clear floats
            elif line.strip().startswith("**") and line.strip().endswith(":**"):
                html_lines.append(f'<p class="section-heading">{processed}</p>')
            else:
                html_lines.append(f"<p>{processed}</p>")
        else:
            close_lists()
            html_lines.append("")

    if in_indented_code:
        html_lines.append("</code></pre>")
    if in_table:
        html_lines.append("</table>")
    close_lists()

    # Restore LaTeX expressions after processing
    result = "\n".join(html_lines)
    result = restore_latex(result, latex_placeholders)
    return result


def generate_html_document(
    md_content: str,
    title: str,
    base_path: Path,
    display_title_html: str | None = None,
    page_break_before_headings: Iterable[str] | None = None,
) -> str:
    """Generate complete HTML document with styling and embedded images."""
    body_html = markdown_to_html(
        md_content,
        base_path,
        display_title_html=display_title_html,
        page_break_before_headings=page_break_before_headings,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- MathJax 3.x configuration for LaTeX rendering -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true,
                packages: {{'[+]': ['ams']}},
                tags: 'ams'
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
    <style>
        @media print {{
            body {{ margin: 0; padding: 0.5in; }}
            @page {{ margin: 0.75in; size: letter; }}
            figure {{ page-break-inside: avoid; }}
        }}

        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.6;
            max-width: 7.5in;
            margin: 0 auto;
            padding: 1in;
            color: #333;
        }}

        h1 {{
            font-size: 18pt;
            margin-top: 0;
            margin-bottom: 12pt;
            text-align: center;
            border-bottom: none;
        }}

        h2 {{
            font-size: 14pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
            border-bottom: 1px solid #ccc;
            padding-bottom: 4pt;
        }}

        h3 {{
            font-size: 12pt;
            margin-top: 18pt;
            margin-bottom: 8pt;
        }}

        h4, h5, h6 {{
            font-size: 11pt;
            margin-top: 14pt;
            margin-bottom: 6pt;
        }}

        p {{
            margin: 0 0 10pt 0;
            text-align: justify;
        }}

        .display-math {{
            margin: 14pt 0;
            overflow-x: auto;
            text-align: center;
        }}

        .page-break {{
            break-before: page;
            page-break-before: always;
            height: 0;
        }}

        mjx-container[jax="CHTML"] {{
            font-size: 108%;
        }}

        mjx-container[jax="CHTML"][display="true"] {{
            margin: 0.9em 0 !important;
            text-align: center !important;
        }}

        table {{
            border-collapse: collapse;
            width: auto;
            margin: 8pt 16pt 8pt 0;
            font-size: 9pt;
            float: left;
            clear: left;
        }}

        table.full-width {{
            width: 100%;
            float: none;
            clear: both;
        }}

        /* Centered tables for wide/important tables */
        table.centered-table {{
            float: none;
            clear: both;
            margin: 12pt auto;
            display: table;
            max-width: 100%;
        }}

        /* Clear floats after sections */
        h2, h3, hr, figure {{
            clear: both;
        }}

        /* Table notes should clear and stay below */
        .table-note {{
            clear: both;
            font-size: 9pt;
            color: #555;
            margin-top: 4pt;
        }}

        /* Section headings (bold labels before tables) should clear floats */
        .section-heading {{
            clear: both;
            margin-top: 12pt;
            margin-bottom: 4pt;
        }}

        th, td {{
            border: 1px solid #999;
            padding: 3px 8px;
            text-align: left;
            white-space: normal;
            overflow-wrap: anywhere;
            vertical-align: top;
        }}

        th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        code {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 1px 4px;
            border-radius: 2px;
        }}

        pre {{
            background-color: #f5f5f5;
            padding: 12px;
            overflow-x: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 9pt;
            line-height: 1.4;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        blockquote {{
            border-left: 3px solid #ccc;
            margin: 12pt 0;
            padding-left: 16px;
            color: #555;
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 24pt 0;
        }}

        li {{
            margin-bottom: 4pt;
        }}

        strong {{
            font-weight: bold;
        }}

        em {{
            font-style: italic;
        }}

        figure {{
            margin: 20pt 0;
            text-align: center;
        }}

        figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}

        figcaption {{
            font-size: 10pt;
            color: #555;
            margin-top: 8pt;
            text-align: justify;
            padding: 0 20pt;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>
"""


def main():
    """Main entry point."""
    docs_dir = Path(__file__).resolve().parents[1] / "_local" / "psc" / "reporting"
    output_dir = docs_dir / "html"
    output_dir.mkdir(exist_ok=True)

    documents = [
        {
            "input": docs_dir / "research_article_draft_v5.1.md",
            "output": output_dir / "Research_Article_v5.1.html",
            "title": "Automated Detection of Research Front Inflections Using Multi-Signal Gradient Boosting",
            "display_title_html": "Automated Detection of Research Front Inflections<br>Using Multi-Signal Gradient Boosting",
            "page_break_before_headings": ["1. Introduction", "References"],
        },
        {
            "input": docs_dir / "SUPPLEMENTAL_METHODS_v5.1.md",
            "output": output_dir / "Supplemental_Methods_v5.1.html",
            "title": "Supplemental Methods: Automated Detection of Research Front Inflections",
            "page_break_before_headings": ["References"],
        }
    ]

    print("=" * 60)
    print("Publication HTML Generation (with embedded figures)")
    print("=" * 60)

    for doc in documents:
        print(f"\nProcessing: {doc['input'].name}")

        if not doc["input"].exists():
            print("  SKIP: File not found")
            continue

        # Read markdown
        with open(doc["input"], encoding="utf-8") as f:
            md_content = f.read()

        # Generate HTML with base path for resolving images
        html_content = generate_html_document(
            md_content,
            doc["title"],
            docs_dir,
            display_title_html=doc.get("display_title_html"),
            page_break_before_headings=doc.get("page_break_before_headings"),
        )

        # Write HTML
        with open(doc["output"], "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  SUCCESS: {doc['output']}")

    print("\n" + "=" * 60)
    print("HTML files generated with embedded figures!")
    print("\nTo convert to PDF:")
    print("  1. Open HTML file in browser")
    print("  2. Press Ctrl+P (or Cmd+P on Mac)")
    print("  3. Select 'Save as PDF'")
    print("  4. Set margins to 0.75in")
    print("=" * 60)


if __name__ == "__main__":
    main()
