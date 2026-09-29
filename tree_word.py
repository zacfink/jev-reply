#!/Users/zacfinkelstein/.venvs/typesafe/bin/python
"""next_word.py, but each word comes from a yes/no question tree instead of one Choice.

Every question in the tree goes out in ONE request (they run in parallel); code then
walks yes/no from the root to a leaf. Still one round trip per word.

    tree_word.py "tell the customer their order is delayed until tomorrow"
"""
import sys, time
from typesafe_sdk import Noul, TypeSafeClient

WORDS = ("your order is has been was shipped delayed cancelled and will arrive until "
         "today tomorrow next week sorry for the inconvenience wait thanks we").split() + ["<end>"]
MAX_WORDS = 15


def build(words):
    """Split in half: {(first half, second half): {"yes": subtree, "no": subtree}}; a leaf is a word."""
    if len(words) == 1:
        return words[0]
    half = len(words) // 2
    return {(tuple(words[:half]), tuple(words[half:])): {"yes": build(words[:half]), "no": build(words[half:])}}


def questions(tree, out):
    # Each node is conditional on its parents: "IF it's one of these, is it in the first half?"
    if isinstance(tree, dict):
        ((a, b), kids), = tree.items()
        out[" ".join(a)] = Noul(instructions=(
            "Consider the best next word of a short, grammatical message that achieves `goal`, "
            f"continuing `so_far` without repeating it. Suppose it must be one of: {', '.join(a + b)}. "
            f"Is it more likely one of [{', '.join(a)}] than one of [{', '.join(b)}]? "
            "(<end> means the message is already complete.)"))
        questions(kids["yes"], out), questions(kids["no"], out)
    return out


def walk(tree, answers, path):
    while isinstance(tree, dict):
        ((group, _), kids), = tree.items()
        yes = answers[" ".join(group)].noul > 0.5
        path.append("y" if yes else "n")
        tree = kids["yes" if yes else "no"]
    return tree


goal = sys.argv[1] if len(sys.argv) > 1 else "tell the customer their order is delayed until tomorrow"
tree = build(WORDS)
qs = questions(tree, {})
sentence, times = [], []
with TypeSafeClient() as client:
    while len(sentence) < MAX_WORDS:
        t = time.perf_counter()
        a = client.system_one(state={"goal": goal, "so_far": " ".join(sentence)}, questions=qs).answers
        times.append((time.perf_counter() - t) * 1000)
        path = []
        word = walk(tree, a, path)
        print(f"  {times[-1]:4.0f} ms  {word:14} path {''.join(path)}")
        if word == "<end>":
            break
        sentence.append(word)

print(f"\n\"{' '.join(sentence)}\"")
print(f"{len(times)} rounds x {len(qs)} questions each, {sum(times)/1000:.1f} s total")
