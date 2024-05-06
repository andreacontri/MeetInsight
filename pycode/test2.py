from typing import List
import re
from transformers import AutoTokenizer

def split_text_into_chunks(text: str, max_tokens: int = 1024) -> List[str]:
    """
    Split a given text into chunks of a maximum number of tokens, without truncating the last sentence.

    Args:
        text (str): The input text to be split.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 1024.

    Returns:
        List[str]: A list of text chunks, where each chunk has a maximum of `max_tokens` tokens, and the last sentence is not truncated.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_sequence_length = tokenizer.model_max_length

    chunks = []
    start = 0
    end = max_sequence_length

    # Regular expression pattern to match sentence boundaries
    sentence_pattern = r'([.!?])\s*(?=[A-Z])'

    while start < len(text):
        chunk = text[start:end]
        tokens = tokenizer.encode(chunk, add_special_tokens=False)

        if len(tokens) > max_tokens:
            # Find the last sentence boundary before reaching the maximum sequence length
            last_sentence_end = re.lastspan(sentence_pattern, chunk)[1]
            if last_sentence_end:
                end = start + last_sentence_end
            else:
                # If no sentence boundary is found, adjust the end index to the maximum length of a single sentence
                end = start + tokenizer.max_len_single_sentence
        else:
            chunks.append(chunk)
            start = end
            end = start + max_sequence_length

    return chunks

# Example usage
long_text = "This is a relatively short sentence. " * 100 + "This is an extremely long sentence that should be placed in its own chunk to avoid truncation because it exceeds the maximum sequence length of the tokenizer."
chunks = split_text_into_chunks(long_text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    print("-" * 50)