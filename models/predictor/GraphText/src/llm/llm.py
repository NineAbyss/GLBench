"""Contains classes for querying large language models. Modified from APE repo"""
from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt, max_new_tokens=1, choice_only=False):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    # @abstractmethod
    # def log_probs(self, text, log_prob_range):
    #     """Returns the log probs of the text.
    #     Parameters:
    #         text: The text to get the log probs of. This can be a string or a list of strings.
    #         log_prob_range: The range of characters within each string to get the log_probs of.
    #             This is a list of tuples of the form (start, end).
    #     Returns:
    #         A list of log probs.
    #     """
    #     pass
