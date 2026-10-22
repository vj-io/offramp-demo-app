"""Summarise a ticket thread so the next agent can pick it up cold."""

from openai import OpenAI

from helphub.models import ThreadMessage

SUMMARY_SYSTEM_PROMPT = """You write handoff notes for HelpHub support agents taking over a \
ticket. Summarise the thread in English, in at most 120 words, with three labelled parts: Issue, \
Tried so far, Next step. Quote order ids and amounts exactly as written. Skip greetings and \
pleasantries. Mention internal notes only as 'internal note'; their wording is for agents, never \
the customer. Use only information explicitly present in the provided transcript. Do not invent, \
infer, or import any facts from outside the transcript. If the thread is empty, state that the \
issue is unknown and the next step is to contact the customer. Output exactly three lines starting \
with 'Issue:', 'Tried so far:', 'Next step:' — no markdown, no extra commentary."""


def summarize_thread(client: OpenAI, messages: list[ThreadMessage]) -> str:
    transcript = "\n".join(f"[{m.sent_at}] {m.role} ({m.author}): {m.text}" for m in messages)
    response = client.chat.completions.create(
        model="nvidia/Nemotron-3_5-Lightning",
        max_tokens=400,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Ticket thread, {len(messages)} messages:\n\n{transcript}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()
