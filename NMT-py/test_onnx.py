import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch
import json

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Arguments")

# 添加位置参数
parser.add_argument('json_file_path', type=str, help='Where is your json_file')
parser.add_argument('onnxmodel_path', type=str, help='Where is your onnx_model')

# 解析参数
args = parser.parse_args()

# 指定所用onnx对应的分词器名称
model_name = 'Helsinki-NLP/opus-mt-en-de'

# 得到分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 得到onnx模型对应的原始模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 初始化decoder输入向量
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])
decoder_input_ids_np = decoder_input_ids.numpy()

# 从命令行读取json文件路径参数
json_file_path = args.json_file_path

# 打开并读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    idx2word = json.load(file)

# 指定onnx模型的路径
onnx_model_path = args.onnxmodel_path

# 加载onnx模型
session = ort.InferenceSession(onnx_model_path)

# 创建示例输入并分词得到输入向量
input_text = "After finishing her work,she decided to go for a walk in the park to relax."
i = tokenizer.tokenize(input_text)
inputs = tokenizer(input_text, return_tensors="np")

# 获取模型输入名称
onnx_inputs = {
    'input_ids': inputs['input_ids'].astype(np.int64),
    'attention_mask': inputs['attention_mask'].astype(np.int64),
    'onnx::Reshape_2': decoder_input_ids_np
}

# 存储模型每一步输出的单词
result = []
epoch = inputs['input_ids'].shape[1] - 1
for _ in range(epoch):
    # 运行模型
    outputs = session.run(None, onnx_inputs)
    word_idx = outputs[0].argmax(axis=-1)[0][0]
    word_idx_str = str(word_idx)
    pre_word = idx2word[word_idx_str]
    word = pre_word[1:] if pre_word[0] == '▁' else pre_word
    result.append(word)
    onnx_inputs = {
        'input_ids': onnx_inputs['input_ids'][:, 1:].astype(np.int64),
        'attention_mask': onnx_inputs['attention_mask'][:, 1:].astype(np.int64),
        'onnx::Reshape_2': decoder_input_ids_np
    }
# 连接result中的单词
delimiter = ' '
result_text = delimiter.join(result)
# 打印运行结果
print(result_text)

