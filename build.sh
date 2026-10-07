#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

git pull --ff-only

cd draft
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
