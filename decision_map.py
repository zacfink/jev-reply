#!/Users/zacfinkelstein/.venvs/typesafe/bin/python
"""Build decision_map.html: the word-pair network `reply` learned from replies.txt.
Each node is a word; an arrow a→b means b followed a in a saved reply. Rerun after teaching it.

    decision_map.py           # from your replies.txt -> decision_map.html (private, not in git)
    decision_map.py --public  # from replies.starter.txt -> docs/index.html (the GitHub Pages copy)
"""
import collections, json, pathlib, sys
from next_word import END, REPLIES, SHORTLIST, START, STARTER, load, pairs

TOP = 90  # words drawn; the side panel still shows the full shortlist
here = pathlib.Path(__file__).resolve().parent
public = "--public" in sys.argv
source = STARTER if public else REPLIES
rows = load(source)
counts = pairs(e["reply"] for e in rows)
freq = collections.Counter()
for a, nxt in counts.items():
    for b, n in nxt.items():
        freq[a] += n
        freq[b] += n
keep = {START, END} | {w for w, _ in freq.most_common(TOP)}
data = {
    "nodes": [{"id": w, "n": freq[w]} for w in keep],
    "links": [{"source": a, "target": b, "n": n}
              for a, nxt in counts.items() if a in keep for b, n in nxt.items() if b in keep],
    "choices": {a: [[b, n] for b, n in nxt.most_common(SHORTLIST)] for a, nxt in counts.items()},
    "replies": len(rows), "words": len(freq), "start": START, "end": END,
}
html = (here / "decision_map.template.html").read_text().replace("__DATA__", json.dumps(data))
out = here / ("docs/index.html" if public else "decision_map.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"wrote {out.relative_to(here)}: {len(keep)} of {len(freq)} words, {len(data['links'])} arrows")
