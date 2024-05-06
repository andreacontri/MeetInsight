import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Define the conversation data
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

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess the conversation data
input_ids = []
attention_masks = []
for turn in conversation:
    encoded_dict = tokenizer.encode_plus(
        turn,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Use the BERT model to extract features
outputs = model(torch.stack(input_ids), attention_mask=torch.stack(attention_masks))
last_hidden_states = outputs.last_hidden_state

# Define a simple sentiment classification model
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)  # 3 classes: negative, neutral, positive

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialize the sentiment classifier
sentiment_model = SentimentClassifier(model)

# Train the sentiment classifier using the BERT features
# (This is a simplified example, and you'll need to implement the training logic) 