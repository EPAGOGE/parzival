# note/

`NOTE.md`  — the working draft, revision 2. Edit here.
`note.tex` — arXiv-ready LaTeX of the same content. **Not compiled**: this machine has
             no TeX toolchain, so it has been structurally verified (balanced
             environments and braces, even inline math, all 17 cross-references and all
             6 citations resolving, all included graphics present) but never built.
             Compile before posting.
`fig/`     — figures as PDF (for LaTeX) and PNG (for reading), generated from logged
             measurements by `../repro/scripts/figures.py`.

## To build

    latexmk -pdf note.tex        # or: pdflatex note.tex  (twice, for cross-refs)

Requires a TeX distribution (MacTeX is ~5 GB; arXiv compiles server-side, so a local
build is for previewing only).

## Before posting

- confirm the author line and add an affiliation (`note.tex`, marked TODO)
- run one compile and check the three figures place sensibly
- `../boussinesq/NOTE_CLAIMS.md` grades every claim; nothing in the note should exceed
  its grade there
