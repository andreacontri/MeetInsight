import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the required NLTK data
# nltk.download('vader_lexicon')

# Load the SpaCy model
# nlp = spacy.load("en_core_web_sm")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a list of pronouns referring to the other person
other_person_pronouns = ["you", "your", "yours", "yourself"]

# Set the boundary score for potential conflicts
boundary_score = -0.1

def sentiment(df):
	# Analyze each turn in the conversation
	sentiment = []
	for turn in df.Text:
		scores = sia.polarity_scores(turn)

		# Check for potential conflicts based on the compound score
		# if scores['compound'] < boundary_score:
		# 	print(f"Potential conflict detected: {turn}")

		# Check for negative sentiment towards the other person
		for pronoun in other_person_pronouns:
			if pronoun in turn.lower():
				if scores['compound'] < boundary_score:
					sentiment.append(f"!!!: {turn}" + "\n" + "\n")
					break

	return sentiment