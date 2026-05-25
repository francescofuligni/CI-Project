.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=report main.tex

clean:
	latexmk -C -outdir=build -jobname=report main.tex
	rm -rf build
