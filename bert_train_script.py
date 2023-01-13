#%%
from bert_trainer import TrainingCongfig
from torch.utils.tensorboard import SummaryWriter

training_config = TrainingCongfig()

train_test=training_config.dataset.dataset['train']
# train_test[1:10]['features']

train_results = training_config.trainer.train()
training_config.trainer.save_model()
training_config.trainer.log_metrics("train", train_results.metrics)
training_config.trainer.save_metrics("train", train_results.metrics)
training_config.trainer.save_state()

# metrics = training_config.trainer.evaluate(training_config.prepared_ds['validation'])
# training_config.trainer.log_metrics("eval", metrics)
# training_config.trainer.save_metrics("eval", metrics)
# %%
