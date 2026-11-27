#!/usr/bin/env bash
# Build the named manuscript (main.pdf) and the anonymised version (main_anonymous.pdf).
set -e
cd "$(dirname "$0")"
run() { pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; }
rm -f anonymous.flag; run
touch anonymous.flag; run; mv main.pdf main_anonymous.pdf
rm -f anonymous.flag; run
grep -E "^!|undefined" main.log || echo "build ok"
