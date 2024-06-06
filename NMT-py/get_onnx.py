import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Arguments")

# 添加位置参数
parser.add_argument('model_storage_location', type=str, help='Where you want to store model')

# 解析参数
args = parser.parse_args()
# 指定模型名称
model_name = 'Helsinki-NLP/opus-mt-en-de'
# 得到分词器
tokenizer = MarianTokenizer.from_pretrained(model_name)
# 得到transformerNMT模型
model = MarianMTModel.from_pretrained(model_name)

# 创建示例输入
example_text = "Hello world"
# 得到分词后输入
inputs = tokenizer(example_text, return_tensors="pt")

# 创建输入初始向量
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])

# 定义输入名称和动态轴
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "output": {0: "batch_size", 1: "sequence_length"}
}

# 转换为ONNX格式
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask'],decoder_input_ids),
    args.model_storage_location,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=11
)