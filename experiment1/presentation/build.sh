#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build case1_slides.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build case1_with_backup.tex
