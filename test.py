import pandas as pd
import numpy as np
# 新建一个dataframe，赋值随机数
df1 = pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(6,4),columns=list('EFGH'))

df = {}
df['df1'] = df1
df['df2'] = df2
print(df['df1'][['A','B']])