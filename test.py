from transformers import BertTokenizer, BertModel, AdamW


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
inputs1=tokenizer.encode("The emotion of # E3 is positive.")
inputs2=tokenizer.encode("The emotion of # E3 is neural.")
inputs3=tokenizer.encode("The emotion of # E3 is negative.")
tokens1 = tokenizer.convert_ids_to_tokens(inputs1)
tokens2 = tokenizer.convert_ids_to_tokens(inputs2) 
tokens3 = tokenizer.convert_ids_to_tokens(inputs3)
outputs1 = model(inputs1)
outputs2 = model(inputs2)
outputs3 = model(inputs3)
print(tokens1,inputs1)
print(tokens2,inputs2)
print(tokens3,inputs3)
