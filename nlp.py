import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')  # 本地文件夹路径
model = BertModel.from_pretrained('./bert-base-chinese')  # 本地文件夹路径

# 提取文本特征的函数
def get_text_features(text):
    # 使用 BERT 分词器对文本进行编码
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # 将编码后的输入数据传入 BERT 模型
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取 [CLS] token 的向量作为文本的嵌入向量（通常用于分类任务）
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # 获取[CLS]的向量
    
    return cls_embedding.flatten()  # 返回一维向量

# 读取 CSV 文件
input_file = '3.csv'  # 输入的 CSV 文件路径
df = pd.read_csv(input_file)

# 为每一行添加 BERT 文本特征列
text_features_list = []

for _, row in df.iterrows():
    text = row['Word']  # 假设 "Word" 列包含了文本
    text_features = get_text_features(text)
    
    # 将文本特征添加到列表
    text_features_list.append(text_features)

# 将文本特征列添加到 DataFrame 中
text_features_df = pd.DataFrame(text_features_list)

# 将新的文本特征添加到原始 DataFrame 中
df = pd.concat([df, text_features_df], axis=1)

# 保存更新后的 CSV 文件
output_file = '4.csv'  # 输出文件路径
df.to_csv(output_file, index=False)

print(f"数据已保存到 {output_file}")