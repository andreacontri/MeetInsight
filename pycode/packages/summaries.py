from typing import List
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
# from chunk_splitter import split_text_into_chunks
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from typing import List
import re

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

def extractive_summarize_chunks(chunks: List[str], sentences_count: int = 1) -> str:
    """
    Summarize each chunk of text using the Luhn heuristic method for extractive summarization and concatenate all summaries into a final summary text.
    
    Args:
        chunks (List[str]): List of text chunks to be summarized.
        sentences_count (int): Number of sentences to include in the summary of each chunk.
    
    Returns:
        str: A concatenated summary of all chunks.
    """
    
    summaries = []
    summarizer = LuhnSummarizer()
    
    for chunk in chunks:
        parser = PlaintextParser.from_string(chunk, Tokenizer("english"))
        summary = summarizer(parser.document, sentences_count)
        
        summarized_text = ' '.join([sentence._text for sentence in summary])
        summaries.append(summarized_text)
    
    # Combine all summaries into one final text
    final_summary = " ".join(summaries)
    return final_summary

def format_vtt_as_dialogue(text):
    """
    Process the content of a VTT file to extract and format dialogue entries. This function identifies
    timestamped dialogue entries with speaker tags and formats them into a readable dialogue format.

    Args:
        text (str): Raw content from a VTT (Web Video Text Tracks) file.

    Returns:
        str: Formatted string representing the dialogues with timestamps and speaker names.
    """
    # Regex to extract dialogues with timestamps and speaker tags from the VTT content
    entries = re.findall(
        r'(\d{2,}:\d{2}:\d{2}\.\d{3}) --> (\d{2,}:\d{2}:\d{2}\.\d{3}) <v ([^>]+)>(.*?)</v>',
        text, re.DOTALL
    )
    
    formatted_text = []
    # Iterate through each extracted dialogue entry
    for start, end, speaker, content in entries:
        # Clean up the dialogue content by removing newline characters and stripping leading/trailing spaces
        content = content.replace('\n', ' ').strip()
        # Format the dialogue entry with speaker name and timestamps
        formatted_text.append(f"{speaker} ({start} to {end}):\n{content}\n")

    # Join all formatted dialogues into a single string with newline separators
    return '\n'.join(formatted_text)
