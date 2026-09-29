# Reply

A terminal chat assistant that talks purely with [Jev](https://docs.typesafe.ai), TypeSafe's
decision model. Jev can't write text, so `reply` never asks it to: it **picks** whole sentences
from replies it has learned, grades its own answers, and asks you to teach it when it doesn't know.
It's naive and curious on purpose.

```
reply                 # chat (sentence mode, the default)
reply "hi jev"        # one reply, then exit
reply -w / -b         # word mode / binary mode (one letter at a time, experimental)
reply -v              # show each pick's timing and confidence
reply --train         # Jev grades every saved reply and sets aside bad ones
```

At "keep?": `y` saves the reply, typing a better one teaches it, Enter lets Jev judge.

## How it decides
Open `docs/index.html` (or your own `decision_map.html`, rebuilt with `decision_map.py`) for the Decisions view: every question
Jev answers per message, including "is this a world fact I should go learn?". The Words view is
the word-pair network learned from `replies.txt`.

## Files
- `next_word.py` — the app (`reply` is a link to it): sentence / word / binary modes, judge, teaching, lookup
- `tools.py` — time, date, math, remember/recall, open sites, search (Gemini + Google, needs `GEMINI_API_KEY`)
- `replies.txt` — everything it can say (`message | reply` per line). Created from
  `replies.starter.txt` on first run and kept out of git, since it fills with your own conversations
- `tree_word.py`, `triage.py` — early experiments
- `test_next_word.py`, `python tools.py` — offline checks (no API calls)

## Setup
Python 3.12 with `typesafe-sdk` (and `google-genai` for search). Set `TYPESAFE_API_KEY`
(and optionally `GEMINI_API_KEY`). The scripts' first line points at a local virtualenv; change it
or run them with your own `python`.
