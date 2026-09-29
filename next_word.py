#!/Users/zacfinkelstein/.venvs/typesafe/bin/python
"""Jev writes a reply one word per round: code holds the word list, Jev picks
the next word. One round trip per word instead of 40 (one per bit).

It learns from replies.txt ("goal | reply" per line): each goal pulls the closest
saved replies as examples and as its word list. Keep or correct each reply and it's
saved for next time.

    reply                                   # keep asking for goals until you quit
    reply "thank them for the meeting link" # one reply, then exit
    reply -v ...                            # show each pick's timing and confidence
    reply -w ...                            # word mode: build the reply one word at a time
    (sentence mode, picking whole saved sentences, is the default; -s still works)
    reply -b ...                            # binary mode: Jev writes the reply one letter at a time
    reply --train                           # Jev grades every saved reply, sets aside bad ones, then practices

Messages that need a tool (time, date, math, remember, recall, open a site) go to tools.py instead.

Pressing enter at "keep?" lets Jev judge the reply: it's saved only if Jev is sure it's good.
"""
import collections, json, os, pathlib, re, sys, time
from typesafe_sdk import Choice, Noul, TypeSafeClient
import tools

REPLIES = pathlib.Path(__file__).resolve().with_name("replies.txt")
START, END = "<start>", "<end>"
PERSONALITY = ("Jev is naive and curious: it's new to the world, loves learning, asks questions about things "
               "it doesn't know, and admits when it doesn't understand instead of pretending. ")
PUNCT = set(".,!?;:")
MAX_WORDS = 25  # stop runaway loops if Jev never picks <end>
SHORTLIST = 20  # words Jev sees per round; well under the API's 255-choice cap
CONTEXT = 24_000  # tokens of saved replies Jev reads per call (its cap is 32k incl. the question)
CANDIDATES = 254  # sentences Jev picks from per round in -s mode (API cap is 255 incl. <end>)
MAX_SENTENCES = 3
REJECTED = REPLIES.with_name("rejected.txt")
KEEP_IF = 0.8  # Jev's p(good reply) needed to save without you
BATCH = 50  # replies graded per request, all in parallel
JUDGE = ("`goal` is either what a reply should say or a message it responds to. Is `reply` a natural, "
         "grammatical reply that a person would actually send for it? Echoing or copying the other "
         "person's words back, broken grammar, and nonsense count as no.")


STARTER = REPLIES.with_name("replies.starter.txt")  # what a fresh copy starts from


def load(path=REPLIES):
    if path == REPLIES and not REPLIES.exists():  # first run: start from the starter replies
        REPLIES.write_text(STARTER.read_text())
    return [dict(zip(("goal", "reply"), map(str.strip, l.split("|", 1))))
            for l in path.read_text().splitlines() if "|" in l]


STOP = set("a an the i you me my your is are am be to of and or it that this for on in at do".split())


def words(text):
    return set(re.findall(r"\w+", text.lower())) - STOP


def lookup(items, query, budget=CONTEXT, text=lambda e: e["goal"] + " " + e["reply"]):
    """The saved items most related to `query` that fit in `budget` tokens (~4 chars each).
    Files can grow forever; each call only reads the best matches. Ties go to newer items."""
    q = words(query)
    ranked = sorted(enumerate(items), key=lambda ie: (-len(q & words(text(ie[1]))), -ie[0]))
    out, used, seen = [], 0, set()
    for _, e in ranked:
        if text(e) in seen:  # repeats waste context
            continue
        seen.add(text(e))
        cost = len(text(e)) // 4 + 8
        if used + cost > budget:
            break
        out.append(e)
        used += cost
    return out


def tokens(text):
    """Words and punctuation as separate tokens: "Thanks, Zac!" -> ["Thanks", ",", "Zac", "!"]."""
    return re.findall(r"[\w'’@&$%/-]+|[.,!?;:]", text)


def untokens(toks):
    """Join tokens back, with no space before punctuation."""
    return re.sub(r" ([.,!?;:])", r"\1", " ".join(toks))


def pairs(replies):
    """Word-pair model: counts[word][next word], over every saved reply."""
    counts = collections.defaultdict(collections.Counter)
    for r in replies:
        words = [START] + tokens(r) + [END]
        for a, b in zip(words, words[1:]):
            counts[a][b] += 1
    return counts


def shortlist(counts, prev, backfill):
    """Likeliest words after `prev`, topped up from `backfill` (the matching examples' words)."""
    ok = (lambda w: w not in PUNCT) if prev in PUNCT else (lambda w: True)  # no "??" or ".,"
    words = [w for w, _ in counts[prev].most_common(SHORTLIST) if ok(w)]
    words += [w for w in backfill if w not in words and ok(w)][:SHORTLIST - len(words)]
    return words + [END] * (END not in words)


def write(client, goal, verbose, history=()):
    everything = load()
    counts = pairs(e["reply"] for e in everything)
    # ponytail: word-overlap lookup, swap for Jev reranking if matches look off
    examples = lookup(everything, goal)
    backfill = list(dict.fromkeys(w for e in examples for w in tokens(e["reply"])))
    sentence, start = [], time.perf_counter()
    while len(sentence) < MAX_WORDS:
        words = shortlist(counts, sentence[-1] if sentence else START, backfill)
        q = {"next": Choice(
            instructions=PERSONALITY + "Pick the next word of a short, grammatical message that achieves `goal`, "
                         "continuing `so_far`. Write it the way the replies in `examples` are written, "
                     "and use `conversation` (what was just said) to understand `goal`. "
                         "Pick <end> once the message is complete.",
            criteria={w: w for w in words if w != END} | {END: "the message is complete; stop"},
        )}
        t = time.perf_counter()
        a = client.system_one(state={"goal": goal, "conversation": convo(history), "examples": examples, "so_far": untokens(sentence)},
                              questions=q).answers["next"]
        if verbose:
            print(f"  {(time.perf_counter() - t) * 1000:4.0f} ms  {a.choice:12} (confidence {a.confidence:.2f})")
        if a.choice == END:
            break
        sentence.append(a.choice)
    reply = untokens(sentence)
    print(f"\n  {reply}\n  ({len(sentence)} tokens, {time.perf_counter() - start:.1f} s)")
    return reply


def opener(sentence):
    """First word, lowercased: two sentences starting the same way in one reply sound repetitive."""
    return (re.findall(r"[\w']+", sentence.lower()) or [""])[0]


def write_sentences(client, goal, verbose, history=()):
    """Sentence mode: Jev picks up to MAX_SENTENCES whole sentences from saved replies."""
    everything = load()
    g = set(re.findall(r"\w+", goal.lower()))
    overlap = lambda text: len(g & set(re.findall(r"\w+", text.lower())))
    pool = {}  # sentence -> best overlap with the goal (its own words + the goal it answered)
    for e in everything:
        for sent in re.split(r"(?<=[.!?])\s+", e["reply"]):
            if sent.strip():
                pool[sent] = max(pool.get(sent, 0), overlap(sent) + overlap(e["goal"]))
    # ponytail: word-overlap retrieval, swap for a Jev relevance pass if good sentences get missed
    candidates = sorted(pool, key=lambda x: -pool[x])[:CANDIDATES]
    examples = lookup(everything, goal, CONTEXT - 20 * CANDIDATES)  # leave room for the candidates
    said = {x for _, r in history[-3:] for x in re.split(r"(?<=[.!?])\s+", r or "")}  # don't repeat myself
    picked, start = [], time.perf_counter()
    while len(picked) < MAX_SENTENCES:
        options = [c for c in candidates if c not in picked and c not in said
                   and opener(c) not in {opener(x) for x in picked}]  # no "Hey! Hey, what's up?"
        q = {"next": Choice(
            instructions=PERSONALITY + "`newest_message` is either what the reply should say, or a message to respond to. Build a short "
                         "reply to it one sentence at a time: pick the sentence that should come next after "
                         "`so_far`, in the style of the replies in `examples`. Reply to `newest_message` itself; "
                         "`earlier_messages` only help with words like 'it' or 'that'. Don't repeat what "
                         "`me` already said. If no sentence fits exactly, pick "
                         "the closest one. Pick <end> once the reply is complete; don't add sentences that "
                         "repeat or drift.",
            # no <end> on the first pick, so there's always a reply
            criteria={f"s{i}": c for i, c in enumerate(options)} | ({END: "the reply is complete; stop"} if picked else {}),
        )}
        t = time.perf_counter()
        a = client.system_one(state={"newest_message": goal, "earlier_messages": convo(history[-3:]), "examples": examples,
                                     "so_far": " ".join(picked)},
                              questions=q).answers["next"]
        choice = END if a.choice == END else options[int(a.choice[1:])]
        if verbose:
            print(f"  {(time.perf_counter() - t) * 1000:4.0f} ms  {choice} (confidence {a.confidence:.2f})")
        if choice == END:
            break
        picked.append(choice)
    reply = " ".join(picked)
    print(f"\n  {reply}\n  ({len(picked)} sentence{"s" * (len(picked) != 1)}, {time.perf_counter() - start:.1f} s)")
    return reply


# Binary mode: Jev picks the reply one character at a time from these symbols.
# Lowercase only: Jev doesn't tell "h" from "H" in these questions, so code adds capitals afterwards.
SYMBOLS = ["<end>", " ", ".", ",", "!", "?", "'", "-"] + list("abcdefghijklmnopqrstuvwxyz0123456789")
MAX_CHARS = 80


def label(c):
    return {"<end>": "<end of reply>", " ": "<space>"}.get(c, c)


def recase(text, replies):
    """Capitals for lowercase text: each word's usual casing in saved replies, then sentence starts and "I"."""
    usual = collections.Counter(w for r in replies for w in re.findall(r"[\w']+", r))
    best = {}
    for w, n in usual.items():
        if n > usual[best.get(w.lower(), w.lower())] or w.lower() not in best:
            best[w.lower()] = w if n >= usual[w.lower()] else w.lower()
    text = re.sub(r"[\w']+", lambda m: best.get(m.group(), m.group()), text)
    text = re.sub(r"\bi\b", "I", text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def write_binary(client, goal, verbose, history=()):
    """Binary mode: Jev writes the reply one letter at a time (one Choice per character).
    The text is just bits underneath; -v shows each letter's 7-bit ASCII code."""
    examples = lookup(load(), goal, budget=6_000)
    vocab = collections.Counter(w for e in load() for w in re.findall(r"[a-z0-9']+", e["reply"].lower()))
    text, start = "", time.perf_counter()
    while len(text) < MAX_CHARS:
        t = time.perf_counter()
        options = SYMBOLS if text else SYMBOLS[1:]  # can't end before saying anything
        part = re.search(r"[a-z0-9']*$", text).group()  # the word being spelled right now
        spell = [w for w, _ in vocab.most_common() if w.startswith(part) and w != part][:40] if part else []
        q = {"next": Choice(
            instructions=PERSONALITY + "Pick the next character of a short, natural reply to `goal` that continues `so_far` "
                         "(so_far may end in the middle of a word), written the way the replies in `examples` "
                         "are written. Everything is lowercase here. `current_word` is the word being spelled "
                         "(empty between words); `words_starting_that_way` are real words it could become, most "
                         "common first. Pick <space> or punctuation when the word is finished, and <end of reply> "
                         "once the reply is complete.",
            criteria={label(c): label(c) for c in options})}
        a = client.system_one(state={"goal": goal, "conversation": convo(history), "examples": examples,
                                     "so_far": text, "current_word": part, "words_starting_that_way": spell},
                              questions=q).answers["next"]
        c = next(c for c in options if label(c) == a.choice)
        if verbose:
            bits = format(ord(c), "07b") if len(c) == 1 else "-------"
            print(f"  {(time.perf_counter() - t) * 1000:4.0f} ms  {bits}  {label(c):15} (confidence {a.confidence:.2f})")
        if c == "<end>":
            break
        text += c
    reply = recase(" ".join(text.split()), [e["reply"] for e in load()])
    print(f"\n  {reply}\n  ({len(text)} letters = {len(text) * 7} bits, {time.perf_counter() - start:.1f} s)")
    return reply


def grade(client, pairs):
    """p(good reply) for each (goal, reply), BATCH at a time in one request each."""
    out = []
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        qs = {f"r{j}": Noul(instructions=JUDGE.replace("`goal`", f"`items[{j}].goal`").replace("`reply`", f"`items[{j}].reply`"))
              for j in range(len(chunk))}
        a = client.system_one(state={"items": [{"goal": g, "reply": r} for g, r in chunk]}, questions=qs).answers
        out += [a[f"r{j}"].noul for j in range(len(chunk))]
    return out


def save(goal, reply):
    with REPLIES.open("a") as f:
        f.write(f"{clean(goal)} | {clean(reply)}\n")


def clean(text):
    """One line, no escape codes (arrow keys) or control characters."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]|[\x00-\x1f\x7f]", " ", text).split())


def teach(client, goal, reply):
    fix = clean(input("  keep? [y / type the right reply / enter to let Jev judge] "))
    if fix.lower() == "y":
        if reply:
            save(goal, reply), print("  saved")
        else:
            print("  nothing to save — type the right reply instead")
    elif fix:
        save(goal, fix), print("  saved")
    elif reply:
        p = grade(client, [(goal, reply)])[0]
        print(f"  Jev: {p:.0%} good — " + ("saved" if p >= KEEP_IF else "not saved"))
        if p >= KEEP_IF:
            save(goal, reply)


def train(client, write_fn):
    """Grade every saved reply, set aside bad ones, then practice every goal and keep what Jev approves."""
    lines = [l for l in REPLIES.read_text().splitlines() if "|" in l]
    pairs = [tuple(map(str.strip, l.split("|", 1))) for l in lines]
    t = time.perf_counter()
    scores = grade(client, pairs)
    good = [l for l, p in zip(lines, scores) if p >= 0.5]
    bad = [f"{l}  # jev {p:.2f}" for l, p in zip(lines, scores) if p < 0.5]
    REPLIES.write_text("\n".join(good) + "\n")
    if bad:
        with REJECTED.open("a") as f:
            f.write("\n".join(bad) + "\n")
    print(f"graded {len(lines)} saved replies in {time.perf_counter() - t:.1f} s: "
          f"kept {len(good)}, moved {len(bad)} to rejected.txt")
    for l in bad[:10]:
        print("  x", l)
    goals = list(dict.fromkeys(g for g, _ in pairs))
    print(f"\npracticing {len(goals)} goals...")
    tries = [(g, write_fn(client, g, False)) for g in goals]
    known = {tuple(map(str.strip, l.split("|", 1))) for l in good}
    new = [(g, r) for (g, r), p in zip(tries, grade(client, tries)) if p >= KEEP_IF and r and (g, r) not in known]
    for g, r in new:
        save(g, r)
    print(f"\nsaved {len(new)} new replies Jev approved")


HISTORY = 10  # recent back-and-forths every question sees
UNSURE_BELOW = 0.7  # Jev's p(good) under which it admits it doesn't know and asks to be taught
UNSURE = [
    "Ooh, I don't know that one yet. Can you teach me?",
    "I'm not sure what to say to that yet. What should I say?",
    "I've never heard of that! What is it?",
    "What does that mean? I'm still learning.",
    "I don't understand yet. Can you explain it to me?",
    "Hmm, I don't know how to answer that. How would you answer it?",
    "That's new to me! Can you tell me more?",
]


def unsure(client, message, history):
    """Jev picks the "I don't know yet" line that fits the message best."""
    a = client.system_one(state={"message": message, "conversation": convo(history)}, questions={"line": Choice(
        instructions=PERSONALITY + "Jev doesn't know how to reply to `message` yet. Which line fits best?",
        criteria={f"u{i}": u for i, u in enumerate(UNSURE)})}).answers["line"]
    return UNSURE[int(a.choice[1:])]
CONVERSATION = REPLIES.with_name("conversation.jsonl")  # every turn, kept across restarts


def load_history():
    if not CONVERSATION.exists():
        return []
    turns = [json.loads(l) for l in CONVERSATION.read_text().splitlines() if l.strip()]
    return [(t["them"], t["me"]) for t in turns[-HISTORY:]]


def convo(history):
    return [{"them": m, "me": r} for m, r in history[-HISTORY:]]


def respond(client, message, write_fn, verbose, history):
    """Use a tool if the message needs one; otherwise write a reply and teach. Adds to `history`."""
    a = client.system_one(state={"message": message, "conversation": convo(history)},
                          questions=tools.questions()).answers
    out = tools.run(client, message, a, history)
    if out is None:
        out = write_fn(client, message, verbose, history)
        p = grade(client, [(message, out)])[0] if out else 0.0
        if p < UNSURE_BELOW:  # nothing saved fits: admit it and ask to be taught
            out = unsure(client, message, history)
            print(f"  (Jev: {p:.0%} sure of that) \n\n  {out}")
            fix = clean(input("  teach me: [type what I should say / enter to skip] "))
            if fix:
                save(message, fix), print("  thanks! I'll remember that")
        else:
            teach(client, message, out)
    else:
        print(f"\n  {out}\n  (tool: {a['tool'].choice}, confidence {a['tool'].confidence:.2f})")
    history.append((message, out))
    with CONVERSATION.open("a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"), "them": message, "me": out}) + "\n")


def main():
    args = sys.argv[1:]
    verbose = "-v" in args
    flags = {"-v", "-s", "-w", "-b", "--train"}
    unknown = [a for a in args if a.startswith("-") and a not in flags]
    if unknown:
        sys.exit(f"unknown option {unknown[0]} — use -w (words), -b (binary), -v (details) or --train")
    write_fn = write_binary if "-b" in args else write if "-w" in args else write_sentences
    goal = clean(" ".join(a for a in args if a not in flags))
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY isn't set — run `source ~/.zshrc` or open a new terminal.")
    history = load_history()  # pick up where the last session left off
    with TypeSafeClient() as client:
        if "--train" in args:
            return train(client, write_fn)
        if goal:
            respond(client, goal, write_fn, verbose, history)
            return
        print("What should the reply say? (blank line or ctrl-c to quit)")
        while goal := input("\n> ").strip():
            respond(client, goal, write_fn, verbose, history)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
