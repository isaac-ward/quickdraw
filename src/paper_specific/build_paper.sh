#!/bin/bash
# Build the paper PDF. texlive lives in the quickdraw container and the paper repo is NOT mounted into
# it, so copy the repo into scratch/ (which is mounted) and run pdflatex there.
#
#     bash src/paper_specific/build_paper.sh
set -e
P=/home/ubuntu/user_irw/icra2027-seamstress
rm -rf /home/ubuntu/user_irw/quickdraw/scratch/paperbuild
cp -r $P /home/ubuntu/user_irw/quickdraw/scratch/paperbuild
cd /home/ubuntu/user_irw/quickdraw/scratch/paperbuild && rm -rf .git
docker exec quickdraw-app-1 bash -lc '
cd /app/scratch/paperbuild && cp root_code.tex root_build.tex   # a COPY, so root_code.tex stays the main file of the repo
pdflatex -interaction=nonstopmode root_build.tex >build1.log 2>&1
bibtex root_build >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode root_build.tex >build2.log 2>&1
pdflatex -interaction=nonstopmode root_build.tex >build.log 2>&1
echo "--- errors ---"; grep -E "^(!|l\.[0-9]+)" build.log | head -20
echo "--- undefined ---"; grep -iE "undefined (control sequence|reference|citation)" build.log | sort -u | head -20
echo "--- overfull >20pt ---"; grep -oE "Overfull \\\\[hv]box \([0-9.]+pt" build.log | sort -u | tail -5
echo "--- pages ---"; grep -oE "\([0-9]+ pages" build.log | tail -1
'
