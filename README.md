# Usage

The training framework is provided in `train.py`. To start training, simply run `launch/train.sh`. Remember to change the `--tokenizer_path` and `--train_data_path` to the paths to your local files.

A recommended project directory structures as following:
```
.
├── data
│   ├── test.json
│   └── train.json
├── launch
│   └── train.sh
├── models
│   ├── debug
│   └── vocab.txt
├── wandb
└── train.py
```

# TODO

- [ ] Add adjustable hyper-parameters.
- [ ] Visualize on W&B.
