from huggingface_hub import InferenceClient


class LocalGenerator:
    """
    Generator that uses HuggingFace Inference API to generate text remotely.
    """

    def __init__(self, model_repo: str = "HuggingFaceH4/zephyr-7b-beta", token: str = '*******'):
        """
        Initialize with HuggingFace model repo and your access token.

        Args:
            model_repo (str): HuggingFace model repo name.
            token (str): Your HuggingFace access token (if needed).
        """
        self.client = InferenceClient(model=model_repo, token=token)

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        """
        Generate answer using the HuggingFace Inference API.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Max number of tokens to generate.

        Returns:
            str: Generated text.
        """
        output = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.3)
        return output
