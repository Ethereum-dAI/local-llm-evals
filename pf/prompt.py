"""promptfoo prompt function.

Returns the fully rendered chat messages (base text from pf/prompt.json plus
the known-token table from datasets/lookup.json). Referenced from
promptfooconfig.yaml as:
    prompts:
      - file://pf/prompt.py:prompt
"""
import json

from wallet_evals.prompt import build_messages


def prompt(context: dict) -> str:
    return json.dumps(build_messages(context["vars"]["user_message"]))
