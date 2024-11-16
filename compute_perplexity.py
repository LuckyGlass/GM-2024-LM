import tqdm
import math
import json
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer
)


@dataclass
class TestArguments:
    model_path: str = field(metadata={'help': "The path to the pretrained model."})
    data_path: str = field(metadata={'help': "The path to the test dataset."})
    
    
class MyDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, path, model_max_length):
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


parser = HfArgumentParser((TestArguments,))
test_args, = parser.parse_args_into_dataclasses()
model = AutoModelForCausalLM.from_pretrained(test_args.model_path, device_map="auto")
# model.eval()
print(model)
tokenizer = AutoTokenizer.from_pretrained(test_args.model_path)
test_dataset = MyDataset(tokenizer, test_args.data_path, model_max_length=1024)
collator = DataCollatorPaddingToMax(tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collator)


nlls = []
with torch.no_grad():
    for sample in tqdm.tqdm(test_dataloader):
        sample['input_ids'] = sample['input_ids'].cuda()
        sample['attention_mask'] = sample['attention_mask'].cuda()
        sample['labels'] = sample['labels'].cuda()
        nll = model(**sample).loss.cpu().item()
        print(nll)
        nlls.append(nll)

print("Average NLL:", sum(nlls) / len(nlls))
print("PPL:", math.exp(sum(nlls) / len(nlls)))
