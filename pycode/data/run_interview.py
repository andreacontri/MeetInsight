import json
import pandas as pd
from scipy import stats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Ensure the required NLTK data is downloaded
nltk.download('vader_lexicon')

# Load the JSON data from a file
with open('data/survey_responses.json', 'r') as file:
    data = json.load(file)

# Initialize lists to store numerical and text data
numeric_fields = [
    "satisfaction_with_transcripts",
    "timeline_feature_usefulness",
    "speaker_identification_accuracy",
    "extractive_summary_helpfulness",
    "abstractive_summary_helpfulness",
    "sentiment_analysis_usefulness",
    "time_saved",
    "interface_user_friendly",
    "processing_reliability",
    "recommend_to_colleague",
    "handling_complex_meetings"
]

text_fields = [
    "missing_features",
    "improvement_suggestions",
    "employer_mandate_thoughts",
    "employee_evaluation_use",
    "privacy_concerns"
]

numeric_data = {field: [] for field in numeric_fields}
text_data = {field: [] for field in text_fields}

# Separate numerical and text data
for response in data['survey_responses']:
    for field in numeric_fields:
        if field in response and isinstance(response[field], int):
            numeric_data[field].append(response[field])
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            text_data[field].append(response[field])

# Convert numeric data to DataFrame for statistical analysis
numeric_df = pd.DataFrame(numeric_data)

# Calculate basic statistics for numeric data
numeric_stats = numeric_df.describe()

# Perform additional statistical tests if needed
# Example: Perform t-test on satisfaction_with_transcripts
satisfaction_ttest = stats.ttest_1samp(numeric_df['satisfaction_with_transcripts'], popmean=5)

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiment of text responses
sentiment_results = {field: [] for field in text_fields}
for field in text_fields:
    for text in text_data[field]:
        sentiment = sid.polarity_scores(text)
        sentiment_results[field].append(sentiment)

# Convert sentiment results to DataFrame for further analysis
sentiment_df = {field: pd.DataFrame(sentiment_results[field]) for field in text_fields}

# Analyze the frequency of text responses
text_frequency = {field: Counter(text_data[field]) for field in text_fields}

# Print numeric statistics
print("Numeric Data Statistics:")
print(numeric_stats)

# Print t-test result
print("\nT-test on 'satisfaction_with_transcripts':")
print(satisfaction_ttest)

# Print sentiment analysis results
print("\nSentiment Analysis Results:")
for field, df in sentiment_df.items():
    print(f"\nField: {field}")
    print(df.describe())

# Print text frequency analysis
print("\nText Frequency Analysis:")
for field, counter in text_frequency.items():
    print(f"\nField: {field}")
    for text, count in counter.items():
        print(f"{text}: {count}")

# Save the results to JSON files
numeric_stats.to_json('numeric_stats.json', indent=4)
with open('sentiment_analysis.json', 'w') as file:
    json.dump({field: df.to_dict() for field, df in sentiment_df.items()}, file, indent=4)
with open('text_frequency.json', 'w') as file:
    json.dump(text_frequency, file, indent=4)
