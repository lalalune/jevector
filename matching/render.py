"""Generate benchmark documentation and the aggregate viewer from saved measurements."""

import html
from pathlib import Path
from matching.production_reporting import sections


def markdown(sections):
    lines = []
    for title, headers, rows, notes in sections:
        if title:
            lines += ["## " + title, ""]
        if headers:
            lines += (
                [
                    "| " + " | ".join(headers) + " |",
                    "|" + "|".join(["---"] * len(headers)) + "|",
                ]
                + ["| " + " | ".join(row) + " |" for row in rows]
                + [""]
            )
        lines += notes + [""]
    return "\n".join(lines)


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + html.escape(x) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def main():
    data = sections()
    report = (
        "# JEVECTOR / PROFILE MATCHING\n\nExplicit profile values, reciprocal requirements, and measured matching results. Results use saved API responses.\n\n"
        + markdown(data)
    )
    Path("docs/benchmarks.md").write_text(report)
    readme = Path("README.md")
    text = readme.read_text()
    start = "<!-- benchmarks:start -->"
    end = "<!-- benchmarks:end -->"
    if start in text:
        a = text.index(start) + len(start)
        b = text.index(end)
        text = (
            text[:a]
            + "\n\n"
            + markdown(
                [
                    ("", headers, rows, [])
                    for title, headers, rows, notes in data
                    if title == "Retrieval comparison"
                ]
            )
            + "\n"
            + text[b:]
        )
        readme.write_text(text)
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>JEVECTOR / PROFILE MATCHING</title><style>
body{font:16px system-ui;color:#172b42;background:#f4f6f9;max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:28px;letter-spacing:.04em}h2{font-size:21px}p,dd{line-height:1.5}.panel{background:white;border:1px solid #d9e1e9;border-radius:8px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px 9px;text-align:left;border-bottom:1px solid #dde4ea}th{font-size:13px;color:#52677b}td:not(:first-child){white-space:nowrap}dt{font-weight:650;margin-top:14px}dd{margin:5px 0}.muted{color:#52677b;font-size:14px}a{color:#245b93}summary{cursor:pointer;font-weight:600}</style></head><body>
<h1>JEVECTOR / PROFILE MATCHING</h1><p>Decision-model vectors → HNSW candidates → reciprocal checks → ranked profiles.</p>
<p class="muted">Compare Clef, Jev, BGE-small 384, BM25, and BM25 + BGE-small on the same profiles and queries. Retrieval scores and rule-filtered results are separate.</p>"""
    for i, (title, headers, rows, notes) in enumerate(data):
        section = '<section class="panel"><h2>' + html.escape(title) + "</h2>"
        if headers:
            section += table(headers, rows)
        section += (
            "".join('<p class="muted">' + html.escape(note) + "</p>" for note in notes)
            + "</section>"
        )
        if title not in ("Retrieval comparison", "Complete pipeline comparison"):
            section = (
                '<details class="panel"><summary>'
                + html.escape(title)
                + "</summary>"
                + section
                + "</details>"
            )
        page += section
    page += '<p><a href="../README.md">Run instructions</a> · <a href="benchmarks.md">Full report</a></p></body></html>'
    Path("docs/index.html").write_text(page)
    print(
        "Generated viewer, README benchmarks, and full reports from measured artifacts."
    )


if __name__ == "__main__":
    main()
