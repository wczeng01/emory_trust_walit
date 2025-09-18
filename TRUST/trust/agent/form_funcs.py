import functools
import re
from collections import defaultdict
from typing import Literal, Any

from data_struct.variable_struct import VariableMeta

registered = defaultdict(dict)


def register(
    func=None, *, module: Literal["shared", "conversation", "assessment"] | None = None
):
    """
    Register a function to be used in the prompt formatter.
    """
    if func is None:
        return functools.partial(register, module=module)

    def wrapper(func):
        if module is None:
            registered["shared"][func.__name__] = func
        else:
            registered[module][func.__name__] = func
        return func

    return wrapper(func)


def form_msg(msg: str | None, role: str):
    if msg:
        msg = re.sub(r"\s+", " ", msg.strip())
        return f"{role}: {msg}\n"
    return ""


@register
def form_history(history: list[tuple]):
    if history:
        prefix = "Interview History:\n"
        turns = "".join(
            [
                form_msg(turn[0], "Clinician") + form_msg(turn[1], "Patient")
                for turn in history
            ]
        )
        return prefix + turns + "\n"
    else:
        return ""


@register(module="conversation")
def form_da_tags(data: dict):
    tags_ref = {
        "GC": "Initiates or concludes the interview",
        "IS": "Ask predefined assessment questions from the clinical protocol",
        "CQ": "Requests additional details or explanation about patient's previous response; or invites patient to respond or elaborate",
        "CA": "Responds to patient questions with clinical information, or procedural guidance",
        "GI": "Offers instructions, educational information, or procedural guidance",
        "ACK": 'Provides brief verbal confirmation of hearing or understanding patient input (e.g., "I see", "Okay")',
        "EMP": 'Demonstrates understanding of patient\'s emotions or experiences (e.g., "That must be difficult")',
        "VAL": "Restates or summarizes patient's input to verify accurate understanding",
    }
    # prefix = "The response should do the followings:\n"
    if "IS" in data["tags"]:
        tags_ref["IS"] = (
            f'Ask the following assessment question, adapting it naturally based on context: "{data["question"]}"'
        )
    out = "\n".join([f"- {tags_ref[tag]}" for tag in data["tags"]])
    # return prefix + out + "\n"
    return out


@register(module="conversation")
def form_questions(question: str | list):
    if isinstance(question, list):
        return "\n".join(question)
    else:
        prefix = "The predefined interview question is: "
        return prefix + question + "\n"


@register(module="assessment")
def form_range(range_dict: dict) -> str:
    range_str = "[" + ", ".join([k for k, _ in range_dict.items()]) + "]"
    return range_str


@register(module="assessment")
def form_choices(range_dict: dict) -> str:
    range_example = "\n".join([f"{k}: {v}" for k, v in range_dict.items()])
    return "\n" + range_example


@register(module="assessment")
def form_attributes(attributes: list) -> str:
    if not attributes:
        return ""
    condition = [
        f"- If {attr['condition']}, the answer should be {attr['score']}."
        for attr in attributes
    ]
    return "\nNote that:\n" + "\n".join(condition)


@register(module="conversation")
def form_IS_questions(questions: list):
    if questions:
        qstring = "\n".join([f"\t- {q.question}" for q in questions])
        return ":\n" + qstring + "\n"
    else:
        return ""


@register(module="assessment")
def form_next_question(questions: list):
    if questions:
        qstring = "\n".join([f"{q.qid}: {q.question}" for q in questions])
        return qstring
    else:
        return ""


@register(module="assessment")
def form_follow_up(questions: list):
    if questions:
        qstring = "\n".join([f"- {q.question}" for q in questions])
        return "Available follow-up questions:\n" + qstring + "\n"
    else:
        return ""


@register(module="conversation")
def form_reference(ref_history: list | str):
    if ref_history:
        if isinstance(ref_history, str):
            prefix = "Reference:\n"
            return prefix + ref_history + "\n"
        elif isinstance(ref_history, list):
            prefix = "Original Transcript:\n"
            turns = "\n".join([f"{turn[0]}: {turn[1]}" for turn in ref_history])
            return prefix + turns + "\n"
        else:
            raise ValueError("Invalid reference history format")
    else:
        return ""


@register
def form_info_enough(var_template: Any) -> str:
    """
    Build the info-enough snippet from the variable template.

    Accepts:
      - an object with `.template` (dict with 'system'/'user' keys),
      - a plain dict with those keys,
      - or a plain string (already-rendered).

    Returns a single string; empty string if nothing usable.
    """
    if var_template is None:
        return ""

    # If already a string, just return it.
    if isinstance(var_template, str):
        return var_template

    # If it's a dict, try to pull out system/user
    if isinstance(var_template, dict):
        sys_t = (var_template.get("system") or "").strip()
        user_t = (var_template.get("user") or "").strip()
        return "\n".join([s for s in (sys_t, user_t) if s]).strip()

    # If it looks like a VarTemplate/PromptTemplate with `.template` dict
    tmpl = getattr(var_template, "template", None)
    if isinstance(tmpl, dict):
        sys_t = (tmpl.get("system") or "").strip()
        user_t = (tmpl.get("user") or "").strip()
        return "\n".join([s for s in (sys_t, user_t) if s]).strip()

    # Fallback: best-effort stringification
    return str(var_template)


if __name__ == "__main__":
    print(form_msg("Hello, how are you?", "User"))
    print(registered)
