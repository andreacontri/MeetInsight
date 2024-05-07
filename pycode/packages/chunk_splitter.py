import nltk
from transformers import BartTokenizer
from typing import List


# Load the BART-large-cnn tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def split_text_into_chunks(text: str, max_tokens: int = 1024) -> List[str]:
    """
    Split a given text into manageable chunks of a maximum number of tokens, while respecting 
    sentence boundaries. This ensures that sentences are not cut off abruptly in the middle, 
    which is crucial for tasks like summarization or translation where context is important.

    Args:
        text (str): The input text to be split.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 1024.

    Returns:
        List[str]: A list of text chunks, each having a maximum of `max_tokens` tokens.
    """
    
    tokens_chunks = []  # Stores the final chunks of tokens
    current_chunk = []  # Temporarily stores tokens until they reach `max_tokens`

    # Tokenize the text into sentences to respect sentence boundaries
    sentences = nltk.sent_tokenize(text)
    
    for sentence in sentences:
        # Tokenize the sentence to count tokens without special tokens
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        
        # If a single sentence is longer than `max_tokens`, split it further
        if len(sentence_tokens) > max_tokens:
            part_size = max_tokens // 2  # Split into parts smaller than max_tokens
            parts = [sentence[i:i + part_size] for i in range(0, len(sentence), part_size)]
            for part in parts:
                part_tokens = tokenizer.encode(part, add_special_tokens=False)
                if len(part_tokens) > max_tokens:
                    continue  # Skip parts still exceeding `max_tokens`
                if len(current_chunk) + len(part_tokens) > max_tokens:
                    # Save the current chunk and start a new one
                    tokens_chunks.append(tokenizer.decode(current_chunk))
                    current_chunk = part_tokens
                else:
                    current_chunk.extend(part_tokens)
        else:
            # Add sentence tokens to current chunk or save the chunk if it exceeds the limit
            if len(current_chunk) + len(sentence_tokens) > max_tokens:
                tokens_chunks.append(tokenizer.decode(current_chunk))
                current_chunk = sentence_tokens
            else:
                current_chunk.extend(sentence_tokens)
    
    # Add the last chunk if it contains any tokens
    if current_chunk:
        tokens_chunks.append(tokenizer.decode(current_chunk))
    
    return tokens_chunks

# FOR DEBUGGING ONLY #################################################################
# Example usage
# vtt_filename = "data/example2.txt"
# with open(vtt_filename, 'r', encoding='utf-8') as file:
#     vtt_content = file.read()

# chunks = split_text_into_chunks(vtt_content)

# print(chunks[1])
