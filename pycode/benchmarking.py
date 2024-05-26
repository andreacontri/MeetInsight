################################################################
from datasets import load_dataset


from datasets import load_dataset
ds = load_dataset("edinburghcstr/ami", "ihm")

print(ds)


# dataset = load_dataset("glue", "mrpc", split="train")

# ################################################################
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def encode(examples):
#     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

# dataset = dataset.map(encode, batched=True)
# # print(dataset[0])

# dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

# ################################################################

# import torch

# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# print()