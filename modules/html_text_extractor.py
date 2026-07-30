"""Utilities for extracting readable plain text from HTML documents."""

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import re
import sys
import warnings


def format_text(text, width=80, indent='    '):
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.strip()
    paragraphs = re.split(r'\n\s*\n+', text)
    formatted = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        joined = ' '.join(lines)
        formatted.append(joined)
    return '\n\n'.join(formatted)


def extract_text_from_html(html, source=None):
    """Extract formatted visible text from an HTML string or readable stream."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, 'html.parser')

    for warning in caught_warnings:
        if issubclass(warning.category, XMLParsedAsHTMLWarning):
            label = source if source is not None else "<html string>"
            print(
                f"Warning: parsed XML-looking document as HTML: {label}",
                file=sys.stderr,
            )
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for paragraph in soup.find_all("p"):
        paragraph.append("\n\n")
    return format_text(soup.get_text())


def extract_text(filename):
    """Extract formatted visible text from an HTML file path."""
    with open(filename, "rb") as f:
        content = f.read()

    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(
            f"Warning: could not decode {filename} as UTF-8 at byte {exc.start}; "
            "using Windows-1252 fallback.",
            file=sys.stderr,
        )
        html = content.decode("cp1252", errors="replace")

    return extract_text_from_html(html, source=filename)


def dump_text(filename):
    """Print extracted text from an HTML file path to stdout."""
    print(extract_text(filename))
