import pandas as pd

# 读取CSV文件
file_path = '5.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 获取最后一列的名称
last_column = df.columns[-1]

# 将最后一列的空缺值替换为98
df[last_column] = df[last_column].fillna(98)

# 保存修改后的CSV文件
output_file_path = '6.csv'  # 替换为你想要的输出文件路径
df.to_csv(output_file_path, index=False)

print(f"处理完成，结果已保存到 {output_file_path}")