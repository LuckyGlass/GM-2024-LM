from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizer, Trainer, TrainingArguments, HfArgumentParser

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained('models/debug')
tokenizer = BertTokenizer(vocab_file="models/vocab.txt")
tokenizer.add_special_tokens(
    {'pad_token': '[PAD]',
     'cls_token': '[BOS]',
     'sep_token': '[EOS]',
     'mask_token': '[MASK]',
     "unk_token": "[UNK]",
     "bos_token": '[BOS]',
     "eos_token": '[EOS]'})

# 将模型设置为评估模式
model.eval()

input_text = "我"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
output = model.generate(
    input_ids,
    max_length=50,  # 输出文本的最大长度
    num_return_sequences=1,  # 返回的文本数量
    no_repeat_ngram_size=2,  # 避免重复的n-gram
    do_sample=True,  # 启用采样
    top_k=50,  # 在采样时只考虑前50个最可能的下一个词
    top_p=0.95,  # 使用核采样
    temperature=1.0,  # 控制输出的随机性
)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
