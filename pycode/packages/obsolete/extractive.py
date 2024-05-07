from chunk_splitter import split_text_into_chunks
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from typing import List
import re
from openai import summarize_text

# nltk.download('punkt')  # make sure the punkt tokenizer is downloaded

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
        print(summarize_text(chunk))
        print()
    
    # Combine all summaries into one final text
    final_summary = " ".join(summaries)
    return final_summary

def format_vtt_as_dialogue(text):
    # Find all dialogue entries with timestamps and speaker tags
    entries = re.findall(r'(\d{2,}:\d{2}:\d{2}\.\d{3}) --> (\d{2,}:\d{2}:\d{2}\.\d{3}) <v ([^>]+)>(.*?)</v>', text, re.DOTALL)
    
    formatted_text = []
    for start, end, speaker, content in entries:
        # Clean up the content
        content = content.replace('\n', ' ').strip()
        formatted_text.append(f"{speaker} ({start} to {end}):\n{content}\n")

    return '\n'.join(formatted_text)

# # Use the split_text_into_chunks function from previous examples to get chunks
vtt_filename = "data/formatted_output.vtt"
with open(vtt_filename, 'r', encoding='utf-8') as file:
    vtt_content = file.read()

chunks = split_text_into_chunks(vtt_content)

# Summarize the chunks
final_summary = extractive_summarize_chunks(chunks)
final_output = format_vtt_as_dialogue(final_summary)
# print(final_output)
