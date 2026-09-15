"""Per-paragraph character budget check for the paper sections.

The lipsum commands each section was drafted around are a CHARACTER budget, so replacement prose has to
be measured the same way: a paragraph is a blank-line-separated run of source lines that is not a comment
and not inside a float or tabular environment. Prints char/word counts per paragraph against the target
that the lipsum call it replaced would have produced ([1-4]=226, [1-8]=398, [1-16]=755).

    python -m paper_specific.budget <file.tex> [target_chars]
"""
import re
import sys

def paragraphs(path):
    depth, buf, out = 0, [], []
    for line in open(path):
        s = line.strip()
        if s.startswith("\\begin{") and not s.startswith("\\begin{document"):
            depth += 1
        if depth > 0:
            if s.startswith("\\end{"):
                depth -= 1
            continue
        if not s or s.startswith("%"):
            if buf:
                out.append(" ".join(buf)); buf = []
            continue
        if re.match(r"\\(section|subsection|subsubsection|label|input|agent|begin|end)", s) and len(s) < 90:
            continue
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return out

def main(path, target=398):
    ps = paragraphs(path)
    tot = 0
    for p in ps:
        n = len(p); tot += n
        flag = "" if abs(n - target) <= 20 else ("  OVER" if n > target else "  under")
        print(f"  {n:4d} vs {target}  ({n - target:+5d}){flag}  {p[:58]}...")
    print(f"  {len(ps)} paragraphs, {tot} chars, {len(' '.join(ps).split())} words")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 398))
