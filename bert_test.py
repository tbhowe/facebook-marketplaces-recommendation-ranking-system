#%%
from transformers import BertTokenizer
from transformers import BertModel
import torch


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'I was born in a water moon. Some people, especially its inhabitants, called it a planet, but as it was only a little over two hundred kilometres in diameter, moon seems the more accurate term.'
# max_length : the maximum length of each sequence. The maximum length of a sequence allowed for BERT is 512

# return_tensors : the type of tensors that will be returned. pt for Pytorch, tf if you use Tensorflow.
bert_input = tokenizer(example_text, padding='max_length', max_length=50,
                       truncation=True, return_tensors="pt")

print(bert_input['input_ids'])

# token_type_ids is a mask that identifies in which sentence the token is. In this case we only have one, so every token is in the same sentence.
print(bert_input['token_type_ids'])
# # BERT adds [CLS], [SEP], and [PAD] to the input, which makes things convenient for us
# # What happens when max_length goes from 15 to 20?
# # What happens when max_length is too small?
example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)

# %%
