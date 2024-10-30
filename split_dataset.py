import json
import numpy as np


np.random.seed(1)


with open('data/train.json', 'r') as f:
    data = json.load(f)
np.random.shuffle(data)
data_dev = data[:3000]
data_train = data[3000:]

with open('data/train_split.json', 'w', encoding='utf8') as f:
    json.dump(data_train, f, ensure_ascii=False, indent=4)
with open('data/dev_split.json', 'w', encoding='utf8') as f:
    json.dump(data_dev, f, ensure_ascii=False, indent=4)
