import os
from time import sleep

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils.basics import logger
from .llm import LLM


class GPT(LLM):
    def __init__(self, openai_name="gpt-3.5-turbo", temperature=0, top_p=1, max_tokens=200, sleep_time=0, **kwargs):
        assert 'OPENAI_API_KEY' in os.environ, 'Please set OPENAI_API_KEY as an environment variable.'
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = openai_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time
        logger.critical(f'Using OPENAI {openai_name.upper()}')
        # logger.critical(f'OPENAI-API-Key= {os.environ["OPENAI_API_KEY"]}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0 if choice_only else self.temperature,
            top_p=self.top_p,
            max_tokens=1 if choice_only else self.max_tokens
        )
        sleep(self.sleep_time)
        return response["choices"][0]["message"]["content"]
