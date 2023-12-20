from abc import ABC, abstractmethod

from huggingface_hub import InferenceClient


class Client(ABC):
    def __init__(self, model, max_new_tokens=512, temperature=1.0, do_sample=False):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    @abstractmethod
    def textgen(self, prompt) -> str:
        pass


class TGIClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = InferenceClient(model=self.model)

    def textgen(self, prompt, **kwargs) -> str:
        # if kwargs not set with default values, set them
        for key, value in {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }.items():
            if key not in kwargs:
                kwargs[key] = value

        return self._client.text_generation(prompt=prompt, **kwargs)


if __name__ == "__main__":
    client = TGIClient(model="http://127.0.0.1:8080")
    output = client.textgen(prompt="Write a code for snake game")
    print(output)
