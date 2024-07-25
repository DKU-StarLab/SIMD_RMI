import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "/home/test/SIMD_RMI/RMI_SIMD/results/rmi_build.csv"
output_dir = "/home/test/SIMD_RMI/RMI_SIMD/results/"

# CSV 파일 읽기
data = pd.read_csv(file_path)

dataset_dict = {
    "books_200M_uint64": "books",
    "fb_200M_uint64": "fb",
    "osm_cellids_200M_uint64": "osmc",
    "wiki_ts_200M_uint64": "wiki"
}
model_dict = {
    "linear_regression": "LR",
    "linear_spline": "LS",
    "linear_regression_welford": "LR_SIMD"
}
data.replace({**dataset_dict, **model_dict}, inplace=True)
data['size_in_MiB'] = data['size_in_bytes'] / (1024 * 1024)
datasets = data['dataset'].unique()
data = data[(data['switch_n'] == 8)]
# 서브플롯 생성
num_datasets = len(datasets)
num_cols = 2  # 한 행에 플롯을 몇 개씩 보여줄 것인지
num_rows = (num_datasets + num_cols - 1) // num_cols  # 필요한 행 수 계산
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8.5, 7.5))  # 전체 그림판 및 축 설정
cs = plt.cm.Dark2.colors
for i, dataset in enumerate(datasets):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col] if num_rows > 1 else axes[col]
    
    dataset_data = data[(data['dataset'] == dataset)]
    
    # linear_regression 데이터
    linear_data = dataset_data[dataset_data['layer2'] == 'LR']
    ax.plot(linear_data['size_in_MiB'], linear_data['build_time']/1_000_000_000, label='RMI', alpha=0.7, marker='^')
    
    # linear_regression_welford 데이터
    linear_welford_data = dataset_data[dataset_data['layer2'] == 'LR_SIMD']
    ax.plot(linear_welford_data['size_in_MiB'], linear_welford_data['build_time']/1_000_000_000, label='pRMI', alpha=0.7, marker='o')
    
    # 그래프 제목 및 축 레이블 설정
    ax.set_title(f'{dataset}', fontsize=16, fontweight='bold')
    if row==1:
        # ax.set_xlabel('# of segment', fontsize=16, fontweight='bold')
        ax.set_xlabel('Index size [MB]', fontsize=16, fontweight='bold')
    if col==0:
        ax.set_ylabel('Build Time (s)', fontsize=16, fontweight='bold')
    
    # x축을 로그 스케일로 설정
    # ax.set_xscale('log', base=2)
    # ax.set_xlim(2**9, 2**27)
    
    ax.set_xscale('log', base=10)
    ax.set_xlim(10**(-3), 10**3)
    
    # y축 단위를 s로 설정
    ax.set_ylim(0, 10)
    ax.ticklabel_format(style='plain', axis='y')
    
    # 범례 표시
    ax.legend(loc='upper left')
    # 그래프에 grid 설정
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

# 빈 서브플롯 제거
for i in range(len(datasets), num_cols * num_rows):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col] if num_rows > 1 else axes[col]
    fig.delaxes(ax)

# 서브플롯 간 간격 조정
plt.tight_layout()

# PDF 파일로 저장
plt.savefig(output_dir + 'simd_build_abs.svg')