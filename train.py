import json
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizer, Trainer, TrainingArguments, HfArgumentParser
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(default=None, metadata={'help': "The path to the training dataset (json) file."})
    eval_data_path: Optional[str] = field(default=None, metadata={'help': "The path to the eval dataset file."})


@dataclass
class ModelArguments:
    tokenizer_path: Optional[str] = field(default=None, metadata={'help': "The path to the vacobulary file."})


class MyDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, path, model_max_length):
        super().__init__()
        with open(path, 'r') as f:
            data = json.load(f)
        self.input_ids = [tokenizer(s['text'], add_special_tokens=True, padding=False, truncation=True, max_length=model_max_length, return_tensors='pt').input_ids[0] for s in data]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'labels': self.input_ids[index],
            'attention_mask': torch.ones_like(self.input_ids[index]),
        }


class DataCollatorPaddingToMax:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # accept: input_ids, attention_mask, labels
        return {
            'input_ids': torch.nn.utils.rnn.pad_sequence([f['input_ids'] for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            'attention_mask': torch.nn.utils.rnn.pad_sequence([f['attention_mask'] for f in features], batch_first=True, padding_value=0),
            'labels': torch.nn.utils.rnn.pad_sequence([f['labels'] for f in features], batch_first=True, padding_value=-100)
        }


parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
training_args, model_args, data_args = parser.parse_args_into_dataclasses()
# Load the tokenizer
tokenizer = BertTokenizer(vocab_file=model_args.tokenizer_path)
tokenizer.add_special_tokens(
    {'pad_token': '[PAD]',
     'cls_token': '[BOS]',
     'sep_token': '[EOS]',
     'mask_token': '[MASK]',
     "unk_token": "[UNK]",
     "bos_token": '[BOS]',
     "eos_token": '[EOS]'})
# Load the GPT2 architecture
config = GPT2Config(
    vocab_size=tokenizer.vocab_size + 1,  # It's very annoying that BertTokenizer assign values starting from 1...
    n_positions=1024,
    n_embd=504,  # n_embd must be divided by n_head, adopt a smaller value to control the size of the model
    n_layer=12,  # default value
    n_head=12,  # default value
    # n_inner is set to the empirical value, 4 * n_emb
    activation_function='gelu_new',  # default value
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)  # It's initialized based on the default hyperparameters in the config.
num_parameters = sum(p.numel() for p in model.parameters())
print(f"#parameters = {num_parameters}")
print(model)
# Load the dataset
train_dataset = MyDataset(tokenizer, data_args.train_data_path, 1024)
eval_dataset = MyDataset(tokenizer, data_args.eval_data_path, 1024)
collator = DataCollatorPaddingToMax(tokenizer)
# Train!
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

trainer.save_model()
