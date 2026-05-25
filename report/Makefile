PDF := build/report.pdf
DOCS_PDF := ../docs/report.pdf

.PHONY: all clean

all: $(DOCS_PDF)

$(PDF): main.tex references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=report main.tex

$(DOCS_PDF): $(PDF)
	mkdir -p ../docs
	cp $(PDF) $(DOCS_PDF)

clean:
	latexmk -C -outdir=build -jobname=report main.tex
	rm -rf build
