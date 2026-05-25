BUILD_DIR ?= /private/tmp/ci-project-report-build
OUTPUT_PDF := ../docs/report.pdf

.PHONY: all report clean

all: report

report:
	mkdir -p $(BUILD_DIR) ../docs
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) -jobname=report main.tex
	cp $(BUILD_DIR)/report.pdf $(OUTPUT_PDF)
	@echo "Generated $(OUTPUT_PDF)"

clean:
	latexmk -C -outdir=$(BUILD_DIR) -jobname=report main.tex
	rm -rf $(BUILD_DIR)
