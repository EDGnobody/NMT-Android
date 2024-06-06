from transformers import AutoTokenizer
import json

# 指定模型名称
model_name = 'Helsinki-NLP/opus-mt-en-de'
# 得到对应模型的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 得到分词器词表
vocab = tokenizer.get_vocab()

# 将字典转换为word2idx和idx2wordjson文件并导出
with open('word2idx.json', 'w', encoding='utf-8') as json_file:
    json.dump(vocab, json_file, ensure_ascii=False, indent=4)

swapped_data = {value: key for key, value in vocab.items()}

# 将交换后的字典转换为json并导出到文件
with open('idx2word.json', 'w', encoding='utf-8') as json_file:
    json.dump(swapped_data, json_file, ensure_ascii=False, indent=4)