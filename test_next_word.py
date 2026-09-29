#!/Users/zacfinkelstein/.venvs/typesafe/bin/python
"""No API calls: checks the word-pair shortlist. Run: test_next_word.py"""
from next_word import END, SHORTLIST, START, pairs, shortlist

c = pairs(["Thanks for the link", "Thanks for the update", "Thanks for your help"])
assert shortlist(c, "for", [])[:2] == ["the", "your"]           # most common first
assert shortlist(c, START, []) == ["Thanks", END]               # always offers <end>
assert "zebra" in shortlist(c, "for", ["zebra"])                # backfill tops it up
assert len(shortlist(c, "nope", [str(i) for i in range(99)])) == SHORTLIST + 1
print("ok")

from next_word import tokens, untokens
assert tokens("Thanks, Zac! I'm here.") == ["Thanks", ",", "Zac", "!", "I'm", "here", "."]
assert untokens(["Thanks", ",", "Zac", "!", "I'm", "here", "."]) == "Thanks, Zac! I'm here."
assert pairs(["Hi there!", "Hi you!"])["!"] == {END: 2}               # "!" is its own node
print("ok punctuation")
assert not set(shortlist(pairs(["Hi? ?"]), "?", ["?", "!", "ok"])) & set("?!")  # no punctuation after punctuation
print("ok no repeats")

from next_word import lookup
items = [{"goal": f"filler {i}", "reply": "x" * 400} for i in range(500)] + \
        [{"goal": "ask about the exam date", "reply": "When is the exam?"}]
got = lookup(items, "when is my exam", budget=2_000)
assert got[0]["reply"] == "When is the exam?"                     # best match first
assert sum(len(e["goal"] + e["reply"]) // 4 + 8 for e in got) <= 2_000  # stays inside the budget
assert len(lookup(items * 3, "exam", budget=10**9)) == len(items)  # repeats kept once
print("ok lookup")

from next_word import clean, SYMBOLS
assert clean("Hello Jev\n\x1b[A\x1b[B ") == "Hello Jev"                      # arrow keys and newlines gone
assert len(set(SYMBOLS)) == len(SYMBOLS) <= 255                           # fits one Jev Choice
from next_word import recase
assert recase("hey zac! i'm jev. how are you?", ["Hey Zac!", "I'm Jev.", "My name is Jev"]) == "Hey Zac! I'm Jev. How are you?"
print("ok binary")

from next_word import opener
assert opener("Hey, what's up?") == opener("Hey!") == "hey" and opener("Can you tell me?") != opener("Sure!")
print("ok openers")
