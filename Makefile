# Makefile for paper_solvers.tex
# Builds the IEEE conference paper "Specialized, General, Parallel:
# A Three-Solver Comparison for Quadrotor MPC".
#
# Usage:
#   make            # full build: pdflatex x2 + bibtex + pdflatex x2
#   make figures    # regenerate figures/three_way_pareto.pdf from JSON
#   make clean      # remove LaTeX build artifacts (keeps the PDF)
#   make distclean  # also remove paper_solvers.pdf

PAPER   := paper_solvers
BIB     := references_solvers
LATEX   := pdflatex -interaction=nonstopmode -halt-on-error
PYTHON  := python3

.PHONY: all figures clean distclean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(BIB).bib figures/three_way_pareto.pdf
	$(LATEX) $(PAPER).tex
	-bibtex $(PAPER)
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex

figures: figures/three_way_pareto.pdf

figures/three_way_pareto.pdf: figures/three_way_pareto.py \
        results/bench_tuned_solvers_legion.json results/bench_all_solvers.json
	$(PYTHON) figures/three_way_pareto.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).toc

distclean: clean
	rm -f $(PAPER).pdf
