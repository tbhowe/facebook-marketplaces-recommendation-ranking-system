from bert_dataset import TextDataset
from datasets import load_dataset
from datasets import load_metric
from datasets import features

from transformers import BertModel
from transformers import BertTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TrainingCongfig:
    def __init__(self):
        self.dataset=TextDataset()
        self.model_path = 'bert-base-uncased'
        self.feature_extractor = BertTokenizer.from_pretrained(self.model_path)
        self.prepared_ds = self.dataset.dataset.with_transform(self.transform)
        self.metric = load_metric("accuracy")
        self.labels = self.dataset.categories
        self.model = BertModel.from_pretrained(  self.model_path,
                                                    num_labels=len(self.labels),
                                                    id2label={str(i): c for i, c in enumerate(self.labels)},
                                                    label2id={c: str(i) for i, c in enumerate(self.labels)}
                                                )