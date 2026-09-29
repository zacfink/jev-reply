"""Tools for `reply`: Jev decides if a message needs one (and which site, speculatively, in the
same request); code does the work. Jev only chooses; it never writes the answer.

    time / date     "what time is it?"          -> It's 10:32 PM.
    math            "what's 12 * 7.5?"          -> 12 * 7.5 = 90
    remember        "remember my exam is Friday" -> saved to memory.txt
    recall          "when is my exam?"          -> Jev picks the saved fact that answers it
    open            "open youtube"              -> opens it in your browser
    search          "look up when reading week is" -> Gemini searches Google and writes facts,
                    Jev grades each one, the good ones go into memory.txt (needs GEMINI_API_KEY)

Every question also sees the recent `conversation`, so "look that up" or "open it" make sense.
"""
import ast, datetime, operator, os, pathlib, re, subprocess
from urllib.parse import urlparse
from typesafe_sdk import Choice, Noul

MEMORY = pathlib.Path(__file__).resolve().with_name("memory.txt")
SITES = {
    "google": "https://google.com", "youtube": "https://youtube.com", "gmail": "https://mail.google.com",
    "calendar": "https://calendar.google.com", "github": "https://github.com", "spotify": "https://open.spotify.com",
    "onq": "https://onq.queensu.ca", "solus": "https://my.queensu.ca", "linkedin": "https://linkedin.com",
    "claude": "https://claude.ai",
}
TOOLS = {
    "none": "Ordinary conversation or a message to reply to; no tool needed.",
    "time": "Asks what time it is right now.",
    "date": "Asks today's date or day of the week.",
    "math": "Asks to calculate an arithmetic expression with numbers.",
    "remember": "Tells me a fact to remember for later (e.g. 'remember that my exam is Friday').",
    "recall": "Asks about something they told me to remember earlier.",
    "open": "Asks to open a website or app.",
    "search": "Asks me to look something up, search online, or find out about something new "
              "(news, facts, how something works) that a saved fact wouldn't cover.",
}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")  # ponytail: alias, pin a version if answers drift
KEEP_FACT = 0.8  # Jev's p(good fact) needed to remember it
LEARN_IF = 0.7  # Jev's p("this needs a fact I don't have") that makes it go look something up


def questions():
    return {
        "tool": Choice(instructions="Which tool, if any, does `message` need? Use `conversation` "
                                    "(the recent messages) to work out what words like 'it' or 'that' mean.",
                       criteria=TOOLS),
        # speculative: only used if tool == "open"; asked now so it costs no extra round trip
        "site": Choice(instructions="If `message` asks to open a website or app, which one?",
                       criteria={k: k for k in SITES} | {"other": "none of these"}),
        # speculative: decides whether Jev should go learn something, even when no tool was asked for
        "learn": Noul(instructions="Is `message` a question asking for a fact about the world: what something "
                                   "is, how something works, or who, when, or where something happened? Chit-chat, "
                                   "feelings, plans, and questions about Zac or Jev themselves count as no."),
    }


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def words_to_math(text, last=None):
    """'add 3 to that' -> '60 + 3' when the last answer was 60; also plus/minus/times/divided by."""
    t = text.lower()
    if last is not None:
        t = re.sub(r"\b(that|it|the answer|the result)\b", last, t)
    t = re.sub(r"\badd (\S+) to (\S+)", r"\2 + \1", t)
    t = re.sub(r"\bsubtract (\S+) from (\S+)", r"\2 - \1", t)
    t = re.sub(r"\b(?:multiply|times) (\S+) by (\S+)", r"\1 * \2", t)
    t = re.sub(r"\bdivide (\S+) by (\S+)", r"\1 / \2", t)
    for word, sym in (("divided by", "/"), ("plus", "+"), ("minus", "-"), ("times", "*"), ("to the power of", "^")):
        t = t.replace(word, sym)
    return t


def calc(text):
    """Evaluate the arithmetic in `text` safely (numbers and + - * / ^ % only)."""
    expr = max(re.findall(r"[\d.\s+\-*/x×^%()]+", text), key=lambda s: len(s.strip()), default="").strip()
    expr = expr.replace("×", "*").replace("x", "*").replace("^", "**")

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(ast.parse(expr, mode="eval").body)
        return f"{expr.replace('**', '^')} = {round(v, 6):g}"
    except (SyntaxError, ValueError, ZeroDivisionError):
        return "I couldn't find a calculation in that."


def facts():
    return [l for l in MEMORY.read_text().splitlines() if l.strip()] if MEMORY.exists() else []


def parse_facts(text):
    """Lines like "- fact" or "1. fact" from Gemini's answer, cleaned up."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip().replace("**", "")
        if len(line) > 15 and not line.endswith(":"):
            out.append(line)
    return out[:5]


def search(client, message, history):
    """Gemini searches Google and writes facts; Jev keeps the good ones in memory."""
    if not os.environ.get("GEMINI_API_KEY"):
        return "I can't search yet: add GEMINI_API_KEY to ~/.zshrc (free key at aistudio.google.com/apikey)."
    from google import genai
    from google.genai import types
    recent = "\n".join(f"them: {m}\nme: {r}" for m, r in history[-5:])
    gemini = genai.Client()  # keep a reference: a throwaway Client closes before the request is sent
    try:
        resp = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Use Google Search to answer this message: {message}\n\nRecent conversation, for context:\n"
                     f"{recent or '(none)'}\n\nReply with 3 to 5 short facts, one per line, each starting with "
                     "'- '. Each fact must make sense on its own later (name the subject; no 'it' or 'they'). "
                     "No other text.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
    except Exception as e:  # bad key, quota, network: say so instead of crashing the chat
        return f"Search failed: {str(e)[:200]}"
    found = parse_facts(resp.text or "")
    if not found:
        return "I searched but didn't find anything clear."
    chunks = (resp.candidates[0].grounding_metadata.grounding_chunks or []) \
        if resp.candidates and resp.candidates[0].grounding_metadata else []
    sources = list(dict.fromkeys(c.web.title or urlparse(c.web.uri).netloc for c in chunks if c.web))[:2]
    qs = {f"f{i}": Noul(instructions=f"Is `facts[{i}]` a clear, specific, standalone fact that helps answer "
                                     "`question`? Vague, off-topic, or self-contradicting facts count as no.")
          for i in range(len(found))}
    a = client.system_one(state={"question": message, "facts": found}, questions=qs).answers
    kept = [f for i, f in enumerate(found) if a[f"f{i}"].noul >= KEEP_FACT]
    tag = f"[learned {datetime.date.today()}" + (f" · {', '.join(sources)}]" if sources else "]")
    with MEMORY.open("a") as f:
        f.writelines(f"{fact} {tag}\n" for fact in kept)
    lines = "\n".join(f"  - {fact}" for fact in kept) or "  (nothing Jev trusted enough to keep)"
    return (f"Here's what I found{' (' + ', '.join(sources) + ')' if sources else ''}:\n{lines}\n"
            f"  (remembered {len(kept)} of {len(found)} facts)")


def run(client, message, answers, history=()):
    """Run the chosen tool; return its reply, or None for ordinary conversation."""
    tool = answers["tool"].choice
    now = datetime.datetime.now()
    if tool == "time":
        return f"It's {now:%-I:%M %p}."
    if tool == "date":
        return f"Today is {now:%A, %B %-d, %Y}."
    if tool == "math":
        prev = [r for _, r in history if r and re.search(r"= -?[\d.]+$", r)]
        last = re.search(r"= (-?[\d.]+)$", prev[-1]).group(1) if prev else None
        return calc(words_to_math(message, last))
    if tool == "remember":
        fact = re.sub(r"(?i)^\s*(please\s+)?(can you\s+)?remember( that)?\s*", "", message).strip()
        with MEMORY.open("a") as f:
            f.write(fact + "\n")
        return f"Got it, I'll remember: {fact}"
    if tool == "recall":
        fact = recall(client, message)
        if fact:
            return fact
        return search(client, message, history) if os.environ.get("GEMINI_API_KEY") else \
            "I don't remember anything about that."
    if tool == "open":
        site = answers["site"].choice
        if site == "other":
            return "I don't know that site yet. Add it to SITES in tools.py."
        subprocess.run(["open", SITES[site]])
        return f"Opening {site}."
    if tool == "search":
        return search(client, message, history)
    if answers["learn"].noul >= LEARN_IF:  # a world fact it doesn't have: learn it
        known = recall(client, message)
        if known:
            return known
        found = search(client, message, history)
        if found.startswith("Here's what I found"):
            return "I didn't know that, so I looked it up! " + found.removeprefix("Here's what I found")
        return "I don't know that yet, and I can't look things up right now. Can you teach me?"
    return None


def recall(client, message):
    """The saved fact that answers `message`, as a reply, or None."""
    from next_word import lookup
    known = lookup(facts(), message, budget=8_000, text=str)[:254]  # best-matching facts, any file size
    if not known:
        return None
    choice = client.system_one(state={"message": message, "facts": known}, questions={"fact": Choice(
        instructions="Which saved fact answers `message`?",
        criteria={f"f{i}": f for i, f in enumerate(known)} | {"none": "none of the facts answer it"})}
    ).answers["fact"].choice
    if choice == "none":
        return None
    fact = known[int(choice[1:])]
    return f"I learned: {fact}" if "[learned " in fact else f"You told me: {fact}"


if __name__ == "__main__":
    assert calc("what's 12 * 7.5?") == "12 * 7.5 = 90"
    assert calc("2^10") == "2^10 = 1024"
    assert calc("(3 + 4) x 2") == "(3 + 4) * 2 = 14"
    assert calc("1/0") == "I couldn't find a calculation in that."
    assert calc("__import__('os')") == "I couldn't find a calculation in that."
    assert calc(words_to_math("now add 3 to that", "60")) == "60 + 3 = 63"
    assert calc(words_to_math("what's 10 divided by 4 plus 1")) == "10 / 4 + 1 = 3.5"
    assert parse_facts("Here you go:\n- **Reading week** at Queen's is in late October.\n2. Fall term ends in December 2026.\n") == \
        ["Reading week at Queen's is in late October.", "Fall term ends in December 2026."]
    print("ok")
