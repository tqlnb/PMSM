import pandas as pd

# 读取两个CSV文件
high_freq_df = pd.read_csv('data/id_iq_vd_vq.csv')  # 替换为你的高频文件路径
low_freq_df = pd.read_csv('data/torque_speed_time.csv')  # 替换为你的低频文件路径

# 设置时间列为索引
high_freq_df.set_index('time', inplace=True)
low_freq_df.set_index('time', inplace=True)

# 将时间列转换为 datetime 类型
high_freq_df.index = pd.to_datetime(high_freq_df.index)
low_freq_df.index = pd.to_datetime(low_freq_df.index)

# 处理重复的时间戳，通过添加毫秒来区分
high_freq_df['unique_time'] = high_freq_df.index
low_freq_df['unique_time'] = low_freq_df.index

# 为重复的时间戳添加毫秒信息
def add_milliseconds_to_duplicates(df):
    df['time_ms'] = df.groupby(df['unique_time']).cumcount()  # 计算重复时间戳的序号
    df['unique_time'] = df['unique_time'].dt.strftime('%Y-%m-%d %H:%M:%S') + '.' + df['time_ms'].astype(str).str.zfill(3)  # 将毫秒加到时间戳中
    df.drop(columns=['time_ms'], inplace=True)
    return df

high_freq_df = add_milliseconds_to_duplicates(high_freq_df)
low_freq_df = add_milliseconds_to_duplicates(low_freq_df)

# 设置新的唯一时间列为索引
high_freq_df.set_index('unique_time', inplace=True)
low_freq_df.set_index('unique_time', inplace=True)

# 找到共同的时间范围
common_index = high_freq_df.index.intersection(low_freq_df.index)

# 对低频数据进行线性插值
low_freq_df_interp = low_freq_df.reindex(high_freq_df.index).interpolate(method='linear')

# 合并数据
merged_df = pd.concat([high_freq_df, low_freq_df_interp], axis=1)

# 保存合并后的数据
merged_df.to_csv('data/merged.csv')  # 保存合并后的数据到文件
