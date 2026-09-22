#!/usr/bin/env bash
# Build an ICLR source archive containing exactly the files the paper compiles
# from -- nothing else.
#
# The file list is DERIVED, not maintained by hand: a full latexmk run records
# every file pdfTeX opened in main.fls, and bibtex records its style and
# database in main.blg. Anything on disk that the build never touched (figure
# alternates, superseded tables, _archive/, editor scratch) is therefore
# excluded automatically and cannot drift back in.
#
# Usage:  ./make_submission_zip.sh [output.zip]
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-ICLR_submission_$(date +%Y-%m-%d).zip}"

echo "==> Building (needed for a current main.fls / main.bbl)"
latexmk -g -pdf -interaction=nonstopmode main.tex >/dev/null

# Fail closed: never ship a source archive whose build has unresolved references.
if grep -qE "LaTeX Warning: (Reference|Citation).*undefined|Package natbib Warning: Citation .* undefined|There were undefined (references|citations)" \
  main.log
then
  echo "ERROR: build has undefined references or citations; not archiving." >&2
  exit 1
fi

echo "==> Resolving file list"
{
  # Everything pdfTeX read from inside this directory.
  awk '/^INPUT /{print substr($0,7)}' main.fls
  # BibTeX's inputs are not in main.fls -- take them from the log it writes.
  awk -F': ' '/The style file:|Database file #/{print $2}' main.blg
  # main.bbl ships so the archive compiles without a bibtex pass.
  echo main.bbl
} | sed 's#^\./##' \
  | grep -v '^/' \
  | grep -vE '^_archive/' \
  | grep -vE '^main\.(aux|log|out|fls|fdb_latexmk|blg|pdf)$' \
  | sort -u > .archive_manifest

MISSING=$(while read -r f; do [ -e "$f" ] || echo "$f"; done < .archive_manifest)
if [ -n "$MISSING" ]; then echo "ERROR: manifest lists missing files:"; echo "$MISSING"; exit 1; fi

rm -f "$OUT"
zip -q "$OUT" -@ < .archive_manifest

TOTAL=$(find . -path ./_archive -prune -o -type f -print | wc -l)
KEPT=$(wc -l < .archive_manifest)
echo "==> $OUT"
echo "    $KEPT files included, $((TOTAL - KEPT)) present-but-unused files excluded"
du -h "$OUT" | cut -f1 | sed 's/^/    size: /'
