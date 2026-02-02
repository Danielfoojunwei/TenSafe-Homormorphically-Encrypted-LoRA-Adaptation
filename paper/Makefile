# Makefile for MOAI-LoRA Paper
# Target: NeurIPS 2026 / ICML 2026

MAIN = moai_lora
FIGURES_DIR = figures
LATEX = pdflatex
BIBTEX = bibtex
LATEXMK = latexmk

# Default target
all: $(MAIN).pdf

# Full build with bibliography
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick build (no bibtex)
quick: $(MAIN).tex
	$(LATEX) $(MAIN)

# Build with latexmk (recommended)
latexmk: $(MAIN).tex
	$(LATEXMK) -pdf $(MAIN)

# Build figures separately
figures:
	cd $(FIGURES_DIR) && $(LATEX) figures.tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f $(FIGURES_DIR)/*.aux $(FIGURES_DIR)/*.log

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf
	rm -f $(FIGURES_DIR)/*.pdf

# Word count (approximate)
wordcount:
	@detex $(MAIN).tex | wc -w

# Check for common issues
lint:
	@echo "Checking for common LaTeX issues..."
	@grep -n "TODO\|FIXME\|XXX" $(MAIN).tex || echo "No TODOs found"
	@grep -n "\\\\cite{}" $(MAIN).tex || echo "No empty citations"
	@grep -n "\\\\ref{}" $(MAIN).tex && echo "Warning: empty refs found" || echo "No empty refs"

# Create submission archive
submission: $(MAIN).pdf
	mkdir -p submission
	cp $(MAIN).tex submission/
	cp $(MAIN).pdf submission/
	cp -r $(FIGURES_DIR) submission/
	cd submission && zip -r ../moai_lora_submission.zip .
	rm -rf submission
	@echo "Created moai_lora_submission.zip"

# Anonymize for double-blind review
anonymize:
	@echo "Checking for identifying information..."
	@grep -in "university\|institute\|lab\|@" $(MAIN).tex | head -20 || echo "No obvious identifiers found"

.PHONY: all quick latexmk figures clean distclean wordcount lint submission anonymize
