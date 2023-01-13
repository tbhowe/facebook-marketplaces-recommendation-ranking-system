#%%
from bert_dataset import TextDataset
from datasets import load_dataset
from datasets import load_metric
from datasets import features

from transformers import BertModel
from transformers import BertTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TrainingCongfig:
    '''class containing the model definition, training arguments, and training-related methods'''
    
    def __init__(self):
        self.dataset=TextDataset()
        self.model_path = 'bert-base-uncased'
        self.feature_extractor = BertTokenizer.from_pretrained(self.model_path)
        self.prepared_ds = self.dataset.dataset.with_transform(self.transform)
        self.metric = load_metric("accuracy")
        self.labels = self.dataset.categories
        self.data_collator=DataCollatorForTokenClassification(tokenizer=self.feature_extractor, padding=True,return_tensors='pt')
        self.model = BertModel.from_pretrained(  self.model_path,
                                                    num_labels=len(self.labels),
                                                    id2label={str(i): c for i, c in enumerate(self.labels)},
                                                    label2id={c: str(i) for i, c in enumerate(self.labels)}
                                                )

        self.training_args = TrainingArguments(  
                                    output_dir="./bert-classifier",
                                    per_device_train_batch_size=16,
                                    evaluation_strategy="steps",
                                    num_train_epochs=6,
                                    # fp16=True,
                                    save_steps=100,
                                    eval_steps=100,
                                    logging_steps=10,
                                    learning_rate=2e-4,
                                    save_total_limit=2,
                                    remove_unused_columns=False,
                                    push_to_hub=False,
                                    report_to='tensorboard',
                                    load_best_model_at_end=True,
                                    )
        self.trainer = Trainer(
                    model=self.model,
                    args=self.training_args,
                    data_collator=self.data_collator,
                    compute_metrics=self.compute_metrics,
                    train_dataset=self.prepared_ds["train"],
                    eval_dataset=self.prepared_ds["validation"],
                    tokenizer=self.feature_extractor,
                    )
     
    def collate_function(self,batch):
        '''collates examples into a batch'''
        label2id={c: str(i) for i, c in enumerate(self.labels)}
        inputs = self.feature_extractor([x for x in batch['description']], padding = "max_length", truncation=True, return_tensors='pt')
        inputs['labels'] = [label2id[label] for label in batch["labels"]]
        return inputs 
        
    
    def transform(self,example_batch):
        
        '''applies the feature extractor to a batch, returing pytorch tensors'''
        label2id={c: str(i) for i, c in enumerate(self.labels)}
        inputs = self.feature_extractor([x for x in example_batch['description']], padding = "max_length", truncation=True, return_tensors='pt')
        # include the labels
        inputs['labels'] = [int(label2id[label]) for label in example_batch["labels"]]
        return inputs

    def compute_metrics(self,p):
        '''method to compute the metrics on the predictions'''
        return self.metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__ == "__main__":
    # test_config=TrainingCongfig()
    # test_raw_ds=test_config.dataset.dataset['train']
    # example_batch=test_raw_ds[1:4]
    # test_transform=test_config.transform(example_batch)
    # print(test_transform)

    test_config=TrainingCongfig()
    # print(test_config.dataset.dataset['train'])
    test_prepared_ds=test_config.prepared_ds['train'][1]
    print(type(test_prepared_ds['labels']))
    # test_collate=test_config.data_collator(test_prepared_ds[1:4])

    # test_train_ds=test_config.prepared_ds['train']
    # print(test_train_ds['features'])

# %%
