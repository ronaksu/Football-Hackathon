"""Render the technical document to PDF, for a submission form that asks for one.

    python submit/md_to_pdf.py TECHNICAL_DOCUMENT.md "submit/out/Average Guys.pdf"

Markdown to HTML to PDF, printed by headless Chrome or Edge. Deliberately small: it handles the
constructs this project's documents actually use -- headings, tables, fenced code, lists,
blockquotes, rules, and inline bold / italic / code / links -- and nothing else. A general
converter is a dependency; this is thirty lines of regex and a browser that is already installed.

The submission form accepts Markdown as well as PDF, so this is a convenience rather than a
requirement. Nothing in the pipeline depends on it.
"""

import argparse
import html
import os
import re
import shutil
import subprocess
import sys

CSS = """
@page { size: A4; margin: 17mm 15mm; }
body { font: 10.5pt/1.5 "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #1b2430;
       max-width: 100%; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 4pt; letter-spacing: -0.3pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1px solid #d6dde6; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #2b3a4d; page-break-after: avoid; }
p, li { margin: 0 0 6pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
code { font: 9pt/1.4 Consolas, "SF Mono", monospace; background: #f1f4f8;
       border: 1px solid #e0e6ed; border-radius: 2px; padding: 0 3px; }
pre { background: #f7f9fb; border: 1px solid #e0e6ed; border-left: 3px solid #7d8fa3;
      border-radius: 2px; padding: 7pt 9pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; border: 0; padding: 0; font-size: 8.8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d6dde6; padding: 4pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
blockquote { margin: 8pt 0; padding: 4pt 0 4pt 10pt; border-left: 3px solid #c3cdd9;
             color: #44546a; }
hr { border: 0; border-top: 1px solid #d6dde6; margin: 16pt 0; }
a { color: #1b4d7a; text-decoration: none; }
strong { font-weight: 650; }
"""


def inline(text):
    """Inline markdown. Code spans are extracted first so nothing formats inside them."""
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return "\x00{}\x00".format(len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = text.replace("&amp;mdash;", "&mdash;").replace("&amp;nbsp;", "&nbsp;")
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: "<code>{}</code>".format(html.escape(spans[int(m.group(1))])), text)


def render(md):
    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):                                    # fenced code
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            out.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(body))))
            i += 1
            continue

        if re.match(r"^ *\|", line):                                  # table
            rows = []
            while i < len(lines) and re.match(r"^ *\|", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[1:]
            if body and all(re.fullmatch(r":?-{2,}:?", c) for c in body[0] if c):
                body = body[1:]
            out.append("<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
                "".join("<th>{}</th>".format(inline(c)) for c in head),
                "".join("<tr>{}</tr>".format(
                    "".join("<td>{}</td>".format(inline(c)) for c in r)) for r in body)))
            continue

        heading = re.match(r"^(#{1,4}) +(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append("<h{0}>{1}</h{0}>".format(level, inline(heading.group(2))))
            i += 1
            continue

        if re.match(r"^---+ *$", line):
            out.append("<hr>")
            i += 1
            continue

        if re.match(r"^ *[-*] +", line) or re.match(r"^ *\d+\. +", line):
            ordered = bool(re.match(r"^ *\d+\. +", line))
            items = []
            while i < len(lines) and (re.match(r"^ *[-*] +", lines[i])
                                      or re.match(r"^ *\d+\. +", lines[i])
                                      or (items and lines[i].startswith("  ")
                                          and lines[i].strip())):
                if lines[i].startswith("  ") and not re.match(r"^ *([-*]|\d+\.) +", lines[i]):
                    items[-1] += " " + lines[i].strip()
                else:
                    items.append(re.sub(r"^ *([-*]|\d+\.) +", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append("<{0}>{1}</{0}>".format(
                tag, "".join("<li>{}</li>".format(inline(x)) for x in items)))
            continue

        if line.startswith(">"):
            body = []
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>{}</blockquote>".format(inline(" ".join(body))))
            continue

        if not line.strip():
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4} |```| *\||---+ *$|>| *[-*] | *\d+\. )", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>{}</p>".format(inline(" ".join(para))))

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<style>{}</style></head><body>{}</body></html>".format(CSS, "\n".join(out)))


def find_browser():
    for name in ("chrome", "msedge", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        page = render(handle.read())
    tmp = os.path.splitext(args.output)[0] + ".html"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(page)

    browser = find_browser()
    if not browser:
        raise SystemExit("no Chrome or Edge found. The HTML is at {} -- open it and print to "
                         "PDF, or submit the Markdown, which the form also accepts.".format(tmp))

    done = subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                           "--print-to-pdf=" + os.path.abspath(args.output),
                           "file:///" + os.path.abspath(tmp).replace("\\", "/")],
                          capture_output=True, text=True, timeout=180)
    if not os.path.exists(args.output):
        sys.stderr.write(done.stderr[-2000:] + "\n")
        raise SystemExit("the browser did not write a PDF. The HTML is at {}".format(tmp))
    if not args.keep_html:
        os.remove(tmp)
    print("{}  ({:,} bytes)".format(args.output, os.path.getsize(args.output)))


if __name__ == "__main__":
    main()
