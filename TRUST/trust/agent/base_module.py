import abc
import logging
import sys


from llms.prompt_formatter import (
    Patterns,
    PromptData,
)
from trust.llms.gpt_call import ClaudeAgent, GptAgent
from trust.utils.helpers import get_key

logger = logging.getLogger(__name__)


class LLMModule(abc.ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.model = self._load_model()
        self.prompt_formatter = self.model.get_prompt_formatter()

    def _load_model(self):
        # OpenAI
        api_key = get_key(filename="openai", keyname="api_key")
        # org is optional; include only if you have one
        try:
            organization = get_key(filename="openai", keyname="organization")
        except Exception:
            organization = None

        return GptAgent(
            api_key=api_key,
            organization=organization,
            model_name=self.model_name,   # <-- new
            request_url="https://api.openai.com/v1/chat/completions",
        )

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> dict | str | None:
        pass

    def format_prompt(
        self, prompt_templates: dict, params: dict, **pattern_pairs
    ) -> dict:
        """
        Format the prompt using the given template and prompt parameters

        Parameters:
            prompt_templates (dict): Conversation templates
            params (dict): Parameters for the prompt
            pattern_pairs (dict): Pattern pairs for the prompt

        Returns:
            dict: Formatted prompt
        """
        pattern_dict = Patterns()
        for role, template in prompt_templates.items():
            if "{" in template and role != "assistant":
                pattern_dict.fill_patterns(role, template, **pattern_pairs)
        prompt_data = PromptData().from_patterns(
            prompt_templates=prompt_templates, patterns=pattern_dict
        )
        prompts = self.prompt_formatter.format(prompt_data, **params)
        # logger.info(prompts)
        return prompts[0]
