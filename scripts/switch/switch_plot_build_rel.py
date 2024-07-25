import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

file_path = "/home/test/SIMD_RMI/RMI_SIMD/results/rmi_build.csv"
output_dir = "/home/test/SIMD_RMI/RMI_SIMD/results/"

# CSV 파일 읽기
data = pd.read_csv(file_path)
data = data.groupby(['dataset','layer1','layer2','n_models','size_in_bytes','switch_n']).mean().reset_index()

dataset_dict = {
    "books_200M_uint64": "books",
    "fb_200M_uint64": "fb",
    "osm_cellids_200M_uint64": "osmc",
    "wiki_ts_200M_uint64": "wiki"
}
model_dict = {
    "linear_regression": "LR",
    "linear_spline": "LS",
    "linear_regression_welford": "pRMI"
}
data.replace({**dataset_dict, **model_dict}, inplace=True)
data['size_in_MiB'] = data['size_in_bytes'] / (1024 * 1024)
datasets = data['dataset'].unique()

def calculate_change_ratio(group):
    base_time = group[group['switch_n'] == 8]['build_time'].values[0]
    group['build_time_ratio'] = group['build_time'] / base_time
    return group

data = data[(data['dataset'] == 'books')]
data = data[(data['layer2'] == 'pRMI')]

df_grouped = data.groupby(['n_models','dataset']).apply(calculate_change_ratio)

df_ratio = df_grouped[['dataset', 'layer1', 'layer2', 'n_models', 'size_in_MiB', 'switch_n', 'build_time_ratio']]

# df_ratio['n_models_log'] = np.log2(df_ratio['n_models'])
df_ratio['size_in_MiB_log'] = np.log10(df_ratio['size_in_MiB'])
models_size = df_ratio['size_in_MiB_log']
switch_n = df_ratio['switch_n']
build_time_ratio = df_ratio['build_time_ratio']

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# 데이터 플롯
# ax.scatter(n_models, switch_n, build_time_ratio, c='r', marker='o')
# ax.plot_surface(n_models, switch_n, build_time_ratio)
ax.plot_trisurf(models_size, switch_n, build_time_ratio, cmap='viridis', edgecolor='none')

ax.set_zlim(0.5, 2)

yticks = np.arange(8, max(switch_n)+1, 8)
ax.set_yticks(yticks)

# 축 라벨링
ax.set_xlabel('Index size [MB](10^N)')
ax.set_ylabel('switch_n')
ax.set_zlabel('build_time_ratio')

plt.savefig(output_dir + 'switch_build_rel.svg')