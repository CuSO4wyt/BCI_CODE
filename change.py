import pandas as pd
import opencc

# 繁体字转换为简体字的函数
def convert_to_simplified(text):
    converter = opencc.OpenCC('t2s.json')  # 从繁体到简体
    return converter.convert(text)

# 读取 CSV 文件
input_file = 'zyl.csv'  # 输入的 CSV 文件路径
output_file = 'output.csv'  # 输出的 CSV 文件路径

# 加载 CSV 数据
df = pd.read_csv(input_file)

# 确保 'Word' 列存在，并转换繁体字为简体字
if 'Word' in df.columns:
    df['Word'] = df['Word'].apply(lambda x: convert_to_simplified(x))  # 转换繁体字为简体字
else:
    print("CSV 文件中没有 'Word' 列")

# 将转换后的数据保存为新的 CSV 文件
df.to_csv(output_file, index=False)
print(f"转换后的数据已保存到 {output_file}")

