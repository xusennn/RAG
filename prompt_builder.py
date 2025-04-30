from typing import List

def build_prompt(contexts: List[str], question: str) -> str:
    """
    Assemble a prompt for generation based on retrieved contexts and user question.

    Args:
        contexts (List[str]): List of retrieved documents.
        question (str): User's original question.

    Returns:
        str: Final prompt for generation.
    """
    context_text = "\n\n".join(contexts)
    prompt = (
        f"You are a knowledgeable travel assistant.\n\n"
        f"Based on the following information:\n\n"
        f"{context_text}\n\n"
        f"Please answer the user's question below in a concise and informative manner.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt
