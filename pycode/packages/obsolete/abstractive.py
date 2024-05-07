from typing import List
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
from torch.utils.data import DataLoader
from chunk_splitter import split_text_into_chunks

# Load the BART-large-cnn model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

def abstractive_summarize_chunks(chunks: List[str]) -> str:
    """
    Summarize each chunk of text using a pretrained model and concatenate all summaries
    into a final summary text.

    Args:
        chunks (List[str]): List of text chunks to be summarized.

    Returns:
        str: A concatenated summary of all chunks.
    """
    # Determine the device to use based on availability of CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    summaries = []
    
    for chunk in chunks:
        # Prepare the chunk for the model input
        inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(device)
        
        # Generate a summary with specific configuration
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the generated tokens to a summary string, omitting special tokens
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)
    
    # Concatenate all individual summaries into a single text
    final_summary = " ".join(summaries)
    return final_summary


def batch_abstractive_summarize_chunks(chunks: List[str], batch_size: int = 8) -> str:
    """
    Summarize text chunks in batches using a pre-trained model and concatenate all batch summaries 
    into a final summary text.

    Args:
        chunks (List[str]): List of text chunks to be summarized.
        batch_size (int): Number of chunks to process in a single batch.

    Returns:
        str: A concatenated summary of all chunks.
    """
    # Determine the device based on CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Encode all chunks into a dataset
    dataset = [tokenizer.encode(chunk, return_tensors="pt", truncation=True, max_length=1024) for chunk in chunks]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    summaries = []

    # Process batches of data
    for batch in data_loader:
        inputs = batch.to(device)
        
        # Use mixed precision if on CUDA for efficiency
        if device == "cuda":
            with torch.cuda.amp.autocast():
                summary_ids = model.generate(
                    inputs,
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=2,
                    early_stopping=True
                )
        else:
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=2,
                early_stopping=True
            )
        
        # Decode generated summaries and add to list
        summary_texts = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        summaries.extend(summary_texts)

    # Concatenate all summaries into one final text
    final_summary = " ".join(summaries)
    return final_summary



# FOR DEBUGGING ONLY #################################################################

# # Use the split_text_into_chunks function from previous examples to get chunks
vtt_filename = "data/formatted_output.vtt"
with open(vtt_filename, 'r', encoding='utf-8') as file:
    vtt_content = file.read()

chunks = split_text_into_chunks(vtt_content)

# Summarize the chunks
final_summary = abstractive_summarize_chunks(chunks)
print(final_summary)

final_summary = batch_abstractive_summarize_chunks(chunks)
print(final_summary)
