#!/Users/zacfinkelstein/.venvs/typesafe/bin/python
"""Multi-question Jev call: one request, five judgments, code decides what happens.

All questions see the same state and run in parallel — asking five costs about
one round trip, so ask everything you might branch on.
"""
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ticket = {
    "customer": {"name": "Dana", "plan": "pro", "tenure_months": 14},
    "subject": "Charged twice this month",
    "messages": [
        {"from": "customer", "text": "I got billed on the 3rd and again on the 11th. "
                                     "I need the second charge refunded before payroll."},
        {"from": "support", "text": "Can you confirm the last four digits of the card?"},
        {"from": "customer", "text": "4471. This is the second time this has happened."},
    ],
}

QUESTIONS = {
    "is_billing": Noul(instructions="Is this ticket about a billing or payment problem?"),
    "wants_refund": Noul(instructions="Is the customer explicitly asking for a refund?"),
    "repeat_issue": Noul(
        instructions="Has this customer said the same problem happened to them before?"
    ),
    "tone": Choice(
        instructions="What is the customer's tone across `ticket.messages`?",
        criteria={
            "calm": "States the problem plainly, no time pressure or complaint.",
            "frustrated": "Signals impatience, repetition, or a deadline.",
            "angry": "Hostile, threatens to cancel or escalate.",
        },
    ),
    "urgency": Score(
        instructions="How soon does this need a human response?",
        criteria=[
            "can wait — no deadline, no money at stake",
            "this week — real problem, no immediate consequence",
            "today — money is stuck or a deadline is imminent",
        ],
    ),
}

with TypeSafeClient() as client:
    r = client.system_one(state={"ticket": ticket}, questions=QUESTIONS)

a = r.answers
print(f"model={r.model}  tokens in/out={r.usage.input_tokens}/{r.usage.output_tokens}\n")
for k, v in a.items():
    print(f"  {k:14} {getattr(v, v.type)!r:28} "
          f"{'' if v.type == 'noul' else f'confidence {v.confidence:.2f}'}")

# --- code owns the policy; the model only supplied the judgments ---
billing = a["is_billing"].noul > 0.8
# score is a float over 0..len(levels)-1 (here 0..2), weighted by the distribution
hot = a["urgency"].score >= 1.5 or a["tone"].choice == "angry"
escalate = billing and a["repeat_issue"].noul > 0.6

print("\nroute:", "billing queue" if billing else "general queue",
      "| priority:", "high" if hot else "normal",
      "| escalate to a human:", "yes (repeat billing failure)" if escalate else "no")

if a["tone"].confidence < 0.5:
    print("note: tone was a coin flip — don't let anything user-visible depend on it")
