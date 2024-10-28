from openai import OpenAI
import torch

api_base = "Your API BASE"
api_key = "Your API KEY"

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
    ],
    temperature=0,
)

# print(response.choices[0].message.content)

torch.load("../../datasets/cora.pt")
