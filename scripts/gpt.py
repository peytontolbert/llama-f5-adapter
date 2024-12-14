from openai import OpenAI
client = OpenAI()

class ChatGPT:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def generate(self, system_prompt, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        result = completion.choices[0].message.content
        print(result)
        return result
