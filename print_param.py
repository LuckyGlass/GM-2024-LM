from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('models/vocab.txt')


def f(n_embd, n_layer, n_head):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size + 1,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )
    model = GPT2LMHeadModel(config)  # It's initialized based on the default hyperparameters in the config.
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"#parameters = {num_parameters}")


f(504, 12, 12)
f(612, 8, 12)
f(696, 6, 12)
f(588, 8, 14)
f(672, 6, 16)