import nltk
from transformers import BartTokenizer
from typing import List


nltk.download('punkt')

# Load the BART-large-cnn tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def split_text_into_chunks(text: str, max_tokens: int = 1024) -> List[str]:
    """
    Split a given text into chunks of a maximum number of tokens, respecting sentence boundaries.
    Args:
        text (str): The input text to be split.
        max_tokens (int, optional): The maximum number of tokens per chunk.
    Returns:
        List[str]: A list of text chunks, each having a maximum of `max_tokens` tokens.
    """
    
    tokens_chunks = []
    current_chunk = []
    sentences = nltk.sent_tokenize(text)
    
    for sentence in sentences:
        # Process each sentence, if it's too long, split it further
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_tokens) > max_tokens:
            # If the sentence alone is too long, split into smaller parts
            part_size = max_tokens // 2  # Adjust size as needed
            parts = [sentence[i:i + part_size] for i in range(0, len(sentence), part_size)]
            for part in parts:
                part_tokens = tokenizer.encode(part, add_special_tokens=False)
                if len(part_tokens) > max_tokens:
                    continue  # This part is still too long, consider finer controls here
                if len(current_chunk) + len(part_tokens) > max_tokens:
                    tokens_chunks.append(tokenizer.decode(current_chunk))
                    current_chunk = part_tokens
                else:
                    current_chunk.extend(part_tokens)
        else:
            # Handle normal sentences
            if len(current_chunk) + len(sentence_tokens) > max_tokens:
                tokens_chunks.append(tokenizer.decode(current_chunk))
                current_chunk = sentence_tokens
            else:
                current_chunk.extend(sentence_tokens)
    
    # Add the last chunk if not empty
    if current_chunk:
        tokens_chunks.append(tokenizer.decode(current_chunk))
    
    print()
    print("Splitting into chunks...")
    print(f"Number of chunks: {len(tokens_chunks)}")
    for i, chunk in enumerate(tokens_chunks):
        print(f"Chunk {i+1}: {len(tokenizer.encode(chunk))} tokens")
    print("Chunk splitting completed YEY!!")
    print()
    
    return tokens_chunks

# Example usage
# vtt_filename = "data/example2.txt"
# with open(vtt_filename, 'r', encoding='utf-8') as file:
#     vtt_content = file.read()

# chunks = split_text_into_chunks(vtt_content)

# print(chunks[1])
