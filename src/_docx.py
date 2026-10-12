"""Minimal .docx writer for paragraphs and tables.

Written directly against OOXML because pandoc, LibreOffice and node are not available
in this environment. Scope is deliberately small: headings, body paragraphs and simple
bordered tables, which is all the supplementary tables need.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

import pandas as pd

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="20"/>
</w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/>
<w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""


def _text_run(text: str, bold: bool = False) -> str:
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(str(text))}</w:t></w:r>'


def paragraph(text: str, bold: bool = False, size_half_points: int | None = None) -> str:
    ppr = ""
    if size_half_points:
        ppr = f'<w:pPr><w:rPr><w:sz w:val="{size_half_points}"/></w:rPr></w:pPr>'
    return f"<w:p>{ppr}{_text_run(text, bold)}</w:p>"


def _cell(text: str, width_dxa: int, bold: bool = False, shaded: bool = False) -> str:
    shading = '<w:shd w:val="clear" w:color="auto" w:fill="EFEFEF"/>' if shaded else ""
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width_dxa}" w:type="dxa"/>{shading}</w:tcPr>'
        f"{paragraph(text, bold)}</w:tc>"
    )


def table(frame: pd.DataFrame, total_width_dxa: int = 9360) -> str:
    columns = list(frame.columns)
    n = len(columns)
    base = total_width_dxa // n
    widths = [base] * n
    widths[-1] = total_width_dxa - base * (n - 1)

    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

    header = "<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(
        _cell(col, w, bold=True, shaded=True) for col, w in zip(columns, widths)
    ) + "</w:tr>"

    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for value, w in zip(row, widths):
            text = "" if pd.isna(value) else value
            cells.append(_cell(text, w))
        body.append("<w:tr>" + "".join(cells) + "</w:tr>")

    return (
        f'<w:tbl><w:tblPr><w:tblW w:w="{total_width_dxa}" w:type="dxa"/>'
        f"<w:tblBorders>{borders}</w:tblBorders></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>{header}{''.join(body)}</w:tbl>"
    )


def write_docx(path: Path, body_xml: str, landscape: bool = False) -> None:
    if landscape:
        sect = (
            '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
            '<w:pgMar w:top="850" w:right="850" w:bottom="850" w:left="850" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        )
    else:
        sect = (
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        )

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}{sect}</w:body></w:document>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/_rels/document.xml.rels", DOC_RELS)
        zf.writestr("word/styles.xml", STYLES)
