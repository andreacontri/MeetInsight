from typing import List
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from chunk_splitter import split_text_into_chunks

# Load the BART-large-cnn model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def abstractive_summarize_chunks(chunks: List[str]) -> str:
    """
    Summarize each chunk of text and concatenate all summaries into a final summary text.
    Args:
        chunks (List[str]): List of text chunks to be summarized.
    Returns:
        str: A concatenated summary of all chunks.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    summaries = []
    
    for chunk in chunks:
        # Encode the text for input to BART
        inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(device)
        
        # Generate summary
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        
        # Decode and clean up the summary
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)
    
    # Combine all summaries into one final text
    final_summary = " ".join(summaries)
    return final_summary

import torch
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizerFast

# Load the BART-large model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')


def batch_abstractive_summarize_chunks(chunks: List[str], batch_size: int = 8) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = [tokenizer.encode(chunk, truncation=True, max_length=1024) for chunk in chunks]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    summaries = []
    if torch.cuda.is_available():
        # scaler = torch.cuda.amp.GradScaler()
        for batch in data_loader:
            with torch.cuda.amp.autocast():
                inputs = batch.to(device)
                summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=2, early_stopping=True)
            summary_texts = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            summaries.extend(summary_texts)
    else:
        for batch in data_loader:
            inputs = batch.to(device)
            summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=2, early_stopping=True)
            summary_texts = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            summaries.extend(summary_texts)

    final_summary = " ".join(summaries)
    return final_summary


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
