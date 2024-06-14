from langchain.llms.fake import FakeListLLM

seq_default_list = ['<answer>C</answer>' for _ in range(20)] + ['<answer>?</answer>' for _ in range(200)] + ['<answer>C</answer>' for _ in range(20000)]


class CpuFakeDebugraph_text:
    fake_llm = FakeListLLM(responses=seq_default_list)  # Choose C as Default

    def generate_text(self, prompt, max_new_tokens=1, choice_only=False):
        return self.fake_llm(prompt)[:max_new_tokens]
