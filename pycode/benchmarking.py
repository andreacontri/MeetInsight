import os, sys
import random
import time
import json
import nltk
from transformers import BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import List
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# Import custom modules
from packages.vtt_formatting import format_VTT
from packages.timeline_generator import create_timeline_figure
from packages.stats_generator import create_stats_figure
from packages.chunk_splitter import split_text_into_chunks
from packages.summaries import abstractive_summarize_chunks, extractive_summarize_chunks, format_vtt_as_dialogue
from packages.openai import summarize_text, utility_text
# from packages.sentiment import sentiment

n_transcripts = 20

# Load the BART-large-cnn tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")
# Download the necessary NLTK data files
# nltk.download('punkt')# UNCOMMENT IN DEMO################################



# Sample implementations of the feature functions
def ex_summarization(text, sentences_count: int = 1):
	# Placeholder implementation
	summaries = []
	summarizer = LuhnSummarizer()
	parser = PlaintextParser.from_string(text, Tokenizer("english"))
	summary = summarizer(parser.document, sentences_count)

	summarized_text = ' '.join([sentence._text for sentence in summary])
	summaries.append(summarized_text)
	result = "".join(summaries)
	time.sleep(0.1)
	return result

def ab_summarization(text):
	# Placeholder implementation
	result = abstractive_summarize_chunks(text)
	time.sleep(0.1)
	return result

def ai_summarization(text):
	# Placeholder implementation
	result = summarize_text(text)
	time.sleep(0.1)
	return result

#############################################################################

#############################################################################

def sentiment(text):
	# Initialize the sentiment analyzer
	sia = SentimentIntensityAnalyzer()

	# Define a list of pronouns referring to the other person
	other_person_pronouns = ["you", "your", "yours", "yourself"]

	# Set the boundary score for potential conflicts
	boundary_score = -0.1
	sentiment = []
	# Analyze each turn in the conversation
	def sentiment_f(sentence):
		scores = sia.polarity_scores(sentence)

		# Check for potential conflicts based on the compound score
		# if scores['compound'] < boundary_score:
		# 	print(f"Potential conflict detected: {turn}")

		# Check for negative sentiment towards the other person
		for pronoun in other_person_pronouns:
			if pronoun in sentence:
				if scores['compound'] < boundary_score:
					result = str(f"Negative sentiment towards the other person detected:{sentence}" + "\n" + "\n")
					return result

	# Split text into sentences
	sentences = sent_tokenize(text)

	# Run sentiment analysis on each sentence
	for sentence in sentences:
		result = sentiment_f(sentence)
		if result != None:
			sentiment.append(result)
		# print(f"Sentiment result: {result}")

	time.sleep(0.1)
	return sentiment
	# Placeholder implementation



def utility(text):
	# Placeholder implementation
	result = utility_text(text)
	time.sleep(0.1)
	return result

def key_points(text):
	# Placeholder implementation
	time.sleep(0.1)
	return "key_points_extraction"



# Directory containing the .txt files
txt_dir = 'data/ami-transcripts'

# Get all .txt files in the directory
all_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

# Randomly select n files
selected_files = random.sample(all_files, n_transcripts)

# List to store the results
results = []

# Process each selected file
for file_name in selected_files:
	file_path = os.path.join(txt_dir, file_name)

	with open(file_path, 'r', encoding='utf-8') as file:
		text = file.read()

	# Dictionary to store the results for the current file
	file_result = {
		'file_name': file_name,
		'features': {}
	}

	chunks = split_text_into_chunks(text)

	# Run each feature function and measure the time taken
	for feature_func in [ex_summarization, sentiment]: # , ab_summarization, sentiment, utility,  key_points
		
		start_time = time.time()
		result = feature_func(text)
		end_time = time.time()
		# print(chunks[0])
		# Calculate the time taken and the computational intensity
		time_taken = end_time - start_time
		computational_intensity = len(text) / time_taken if time_taken > 0 else 0
		
		# Store the results
		feature_name = feature_func.__name__
		file_result['features'][feature_name] = {
			'result': result,
			'time_taken': time_taken,
			'computational_intensity': computational_intensity
		}

	# Append the file's results to the overall results list
	results.append(file_result)

# Define the path to the JSON file
output_file_path = os.path.join('data/', 'results.json')

# Load existing data from the JSON file if it exists
if os.path.exists(output_file_path):
	with open(output_file_path, 'r', encoding='utf-8') as output_file:
		contents = output_file.read()
		if contents:
			existing_data = json.loads(contents)
else:
	existing_data = []

# Append the new results to the existing datas
existing_data.extend(results)

# Save the updated results back to the JSON file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
	json.dump(existing_data, output_file, indent=4)

print()




################################################################

# INTERVIEWS 

################################################################


