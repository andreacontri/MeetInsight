import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the required NLTK data
import nltk
nltk.download('vader_lexicon')

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the conversation
conversation = [
    "I really enjoyed the movie we watched last night.",
    "Yeah, it was pretty good, but I didn't like the ending.",
    "What? The ending was the best part! How can you not like it?",
    "I just didn't find it satisfying. It felt rushed and incomplete.",
    "the movie was horrible",
    "Well, you have horrible taste in movies.",
    "I hate you",
    "That was stupid"
]

# Define a list of pronouns referring to the other person
other_person_pronouns = ["you", "your", "yours"]
boundary_score = -0.2

# Analyze each line in the conversation
for line in conversation:
    doc = nlp(line)
    # Analyze the sentiment of the line
    scores = sia.polarity_scores(line)
    # print(f"Line: {line}")
    # print(f"Sentiment scores: {scores}")
    
    # Check for potential conflicts based on the compound score
    if scores['compound'] < boundary_score:
        print(f"Potential conflict detected.: {line}")
    
    # Check for negative sentiment towards the other person
    for pronoun in other_person_pronouns:
        if pronoun in line.lower():
            if scores['compound'] < boundary_score:
                print(f"Negative sentiment towards the other person detected: {line}")
            break
    # for token in doc:

    print()