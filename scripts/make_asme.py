"""Build main_asme.tex (asmejour) from main.tex: same body, ASME header, wide figures as figure*."""
import re, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
main = (d / "main.tex").read_text()
asme = (d / "main_asme.tex").read_text()
head = asme[: asme.index("\\section{Introduction}")]
import re as _re
abs_main = _re.search(r"\\begin\{abstract\}\s*(?:\\noindent\s*)?(.*?)\\end\{abstract\}", main, _re.S).group(1)
head = _re.sub(r"(\\begin\{abstract\}\s*).*?(\\end\{abstract\})", lambda m: m.group(1) + abs_main.strip() + "\n" + m.group(2), head, flags=_re.S)
body = main[main.index("\\section{Introduction}"): main.index("\\bibliographystyle")]
WIDE = ("fig1.pdf", "fig2.pdf", "fig6.pdf")
def widen(m):
    blk = m.group(0)
    if any(w in blk for w in WIDE):
        blk = blk.replace("\\begin{figure}", "\\begin{figure*}").replace("\\end{figure}", "\\end{figure*}")
        blk = blk.replace("width=\\linewidth", "width=0.9\\textwidth")
    return blk
body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", widen, body, flags=re.S)
body = body.replace("\\input{table_taxonomy.tex}", "\\input{table_taxonomy_asme.tex}")
(d / "main_asme.tex").write_text(head + body + "\\bibliographystyle{asmejour}\n\\bibliography{refs}\n\n\\end{document}\n")
print("ok")
