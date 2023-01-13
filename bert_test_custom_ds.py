#%%
from transformers import BertTokenizer
from transformers import BertModel
import torch
from bert_dataset import TextDataset
from transformers import DataCollatorForTokenClassification


def tokenize_function(example):
    return tokenizer(example['description'], truncation=True)

test_dataset=TextDataset()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
inputs={}
inputs['features']=test_dataset.dataset['train'].map(tokenize_function, batched=True)
# inputs['labels']=test_dataset.dataset['train']['labels']
example_input=inputs[1]
print(example_input)


# bert_input = tokenizer(inputs['description'], padding='max_length', max_length=50,
#                        truncation=True, return_tensors="pt")
# package_dict={'features': bert_input , 'labels': inputs['labels']}
# # print(package_dict)
# hs=model(bert_input)


# %%
