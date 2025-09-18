import abc
import json
import logging
import re

import requests

try:
    # Package or module run
    from trust.llms.prompt_formatter import (
        ClaudePromptFormatter,
        GPTPromptFormatter,
        PromptFormatter,
    )
    from trust.utils.helpers import get_key
except ImportError:
    # Relative fallback when executed inside the package
    from .prompt_formatter import (
        ClaudePromptFormatter,
        GPTPromptFormatter,
        PromptFormatter,
    )
    from ..utils.helpers import get_key


logger = logging.getLogger(__name__)


def create_request_header(
    model_name: str, api_key: str, request_url: str, **kwargs
) -> dict:
    if model_name == "claude":
        request_header = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
            if "anthropic-version" not in kwargs
            else kwargs["anthropic-version"],
            "content-type": "application/json",
        }
    elif model_name == "gpt":
        request_header = {"Authorization": f"Bearer {api_key}"}
        # use api-key header for Azure deployments
        if "/deployments" in request_url:
            request_header = {"api-key": f"{api_key}"}
    else:
        raise ValueError(f"Model {model_name} not supported")
    return request_header


def api_endpoint_from_url(request_url) -> str | None:
    """Extract the API endpoint from the request URL."""
    match = re.search(r"^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1] if match else None


class AttemptTracker:
    def __init__(self, max_attempts: int = 5) -> None:
        self.max_attempts: int = max_attempts
        self.failure_count: int = 0

    def __repr__(self) -> str:
        return f"AttemptTracker(max_attempts:{self.max_attempts}, failure_count: {self.failure_count})"

    @property
    def attempts_left(self) -> int:
        return self.max_attempts - self.failure_count

    def record(self) -> None:
        self.failure_count += 1

    def reset(self) -> None:
        self.failure_count = 0

    def block(self) -> bool:
        return self.failure_count >= self.max_attempts


class LLMAgent(abc.ABC):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        request_url: str,
        max_attempts: int = 5,
        **kwargs,
    ):
        self.model_name: str = model_name
        self.api_key: str = api_key
        self.request_url: str = request_url
        self.request_header: dict = create_request_header(
            model_name, api_key, request_url, **kwargs
        )
        self.counter: AttemptTracker = AttemptTracker(max_attempts=max_attempts)

    @staticmethod
    def is_json(data: str) -> bool:
        try:
            _ = json.loads(data)
        except ValueError:
            return False
        return True

    def to_json(self, s: str) -> dict:
        s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if self.is_json(s[: e.pos]):
                return json.loads(s[: e.pos])
            elif e.msg == "Expecting ',' delimiter":
                return self.to_json(s[: e.pos] + "," + s[e.pos :])
            elif e.msg == "Invalid control character at":
                return self.to_json(s[: e.pos] + '",\n' + s[e.pos :])
            else:
                logger.error("Invalid JSON format")
                logger.info(s)
                raise ValueError("Invalid JSON format")

    @abc.abstractmethod
    # trust/llms/gpt_call.py

    def get_response_text(self, response):
        """
        Normalize OpenAI responses (SDK v0 dicts, SDK v1 objects) and surface clear errors.
        """
        # --- OpenAI SDK v1 objects (preferred) ---
        # chat.completions.create(...) returns an object with .choices etc.
        try:
            if hasattr(response, "choices"):
                ch0 = response.choices[0] if response.choices else None
                # For non-streaming:
                if ch0 and hasattr(ch0, "message") and ch0.message and hasattr(ch0.message, "content"):
                    return ch0.message.content
                # For some variants:
                if ch0 and hasattr(ch0, "delta") and ch0.delta and hasattr(ch0.delta, "content"):
                    return ch0.delta.content

            # --- Old dict-shaped responses (SDK v0 style) ---
            if isinstance(response, dict):
                # Standard success shape
                if "choices" in response and response["choices"]:
                    c0 = response["choices"][0]
                    if isinstance(c0, dict):
                        # Non-streaming
                        if "message" in c0 and "content" in c0["message"]:
                            return c0["message"]["content"]
                        # Streaming deltas sometimes look like this
                        if "delta" in c0 and "content" in c0["delta"]:
                            return c0["delta"]["content"]

                # Error payloads like: {"error": {...}}
                if "error" in response:
                    # Surface the actual OpenAI error message
                    err = response["error"]
                    # err can be dict or str
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"OpenAI error: {msg}")

            # If we got here, log and raise
            raise RuntimeError(f"Unexpected OpenAI response shape: {type(response)}: {response}")

        except Exception as e:
            # Re-raise with the raw response to aid debugging
            raise RuntimeError(f"Failed to parse model response: {response}") from e


    @abc.abstractmethod
    def get_prompt_formatter(self) -> PromptFormatter:
        pass

    def generate(self, prompt: dict) -> str | None:
        try:
            if self.counter.block():
                logger.warning(
                    f"Request failed after {self.counter.max_attempts} attempts.\n{prompt}"
                )
            else:
                response = requests.post(
                    self.request_url,
                    headers=self.request_header,
                    json=prompt,
                )
                response = response.json()

                if "error" in response:
                    self.counter.record()
                    logger.warning(
                        f"Request failed with error {response['error']}, {self.counter.attempts_left} attempts remaining."
                    )
                    self.generate(prompt)
                return response
        except Exception as e:
            logger.warning(f"Request failed due to {e}")
        return None

    def call(self, prompt: dict) -> str:
        response = self.generate(prompt)
        response = self.get_response_text(response)
        return response if response else ""


class GptAgent:
    def __init__(
        self,
        api_key: str,
        request_url: str = "https://api.openai.com/v1/chat/completions",
        organization: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        self.api_key = api_key
        self.request_url = request_url
        self.organization = organization
        self.model_name = model_name

    def get_prompt_formatter(self):
        return GPTPromptFormatter()

    def call(self, prompt: dict) -> str:
        import json, requests

        # Make a shallow copy so we don't mutate the caller's dict
        payload = dict(prompt)

        # 🚫 Remove legacy/accidental fields OpenAI doesn't accept
        payload.pop("model_name", None)           # <-- NEW: strip invalid arg
        payload.pop("temperature_top_p", None)    # optional: in case your formatter ever adds this
        payload.pop("organization", None)         # credentials belong in headers only

        # ✅ Ensure "model" is present
        if not payload.get("model"):
            payload["model"] = self.model_name or "gpt-4o"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        resp = requests.post(self.request_url, headers=headers, data=json.dumps(payload), timeout=120)
        data = resp.json()

        if resp.status_code >= 400 or "error" in data:
            raise RuntimeError(f"OpenAI API error: {data.get('error') or data}")

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected OpenAI response: {data}")
    
    def to_json(self, content):
        """
        Best-effort conversion of LLM output to a Python object.
        Accepts dict/list (no-op), or string that may contain
        JSON, code fences, or a 'tags: [...]' style blob.
        """
        # Already structured
        if isinstance(content, (dict, list)):
            return content
        if content is None:
            return {}

        txt = str(content).strip()

        # Strip fenced code, e.g. ```json ... ```
        if txt.startswith("```"):
            txt = re.sub(r"^```[\w-]*\s*|\s*```$", "", txt, flags=re.S)

        # Try to extract the first JSON object/array from the text
        m = re.search(r"(\{.*\}|\[.*\])", txt, flags=re.S)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass  # fall through to heuristics

        # Heuristic: look for a list in a "tags: [ ... ]" style answer
        m = re.search(r"tags?\s*:\s*(\[[^\]]*\])", txt, flags=re.I | re.S)
        if m:
            try:
                arr = json.loads(m.group(1))
                return {"tags": arr}
            except Exception:
                # tolerate non-JSON list like [GC, IS]
                raw = m.group(1).strip("[]")
                items = [x.strip().strip('"\'' ) for x in raw.split(",") if x.strip()]
                return {"tags": items}

        # Last resort: wrap as text
        return {"text": txt}


class ClaudeAgent(LLMAgent):
    def __init__(
        self,
        api_key: str,
        request_url: str = "https://api.anthropic.com/v1/messages",
    ):
        super().__init__("claude", api_key, request_url)

    def get_response_text(self, response) -> str:
        return response["content"][0]["text"]

    def get_prompt_formatter(self) -> ClaudePromptFormatter:
        return ClaudePromptFormatter()


if __name__ == "__main__":
    from ..utils.helpers import get_key
    api_key = get_key(filename="openai", keyname="api_key")
    try:
        org = get_key(filename="openai", keyname="organization")
    except Exception:
        org = None

    gpt_agent = GptAgent(api_key=api_key, organization=org)
    test_sample = {
        "model": "gpt-4o-mini",
        "max_tokens": 60,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi in one word."},
        ],
    }
    print(gpt_agent.call(prompt=test_sample))

    # claude_agent = ClaudeAgent(api_key=api_key)
    # test_sample = {
    #     "model": "claude-3-5-sonnet-20241022",
    #     "max_tokens": 100,
    #     "system": "You are a helpful assistant.",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "What is the capital of France? Return the answer in a JSON object.",
    #         },
    #     ],
    # }
    # print(claude_agent.call(prompt=test_sample))
