#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算两个文本文件的余弦相似度，并把两条曲线画在一张图里
用法示例
    python cosim_plot.py --file1 a.txt --file2 b.txt          # 直接弹窗
    python cosim_plot.py --file1 a.txt --file2 b.txt -o fig.png  # 保存到文件
    python cosim_plot.py --file1 a.txt --file2 b.txt -t cut      # ⻓度不一致截断
"""
import argparse
import math
import os
import sys
from typing import List

import matplotlib.pyplot as plt


def load_vector(path: str) -> List[float]:
    """读取文件，将每行解析为 float，忽略空行与 # 注释。"""
    vec = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                vec.append(float(line))
            except ValueError:
                sys.exit(f'解析 {path} 第 {lineno} 行失败: “{line}” 不是合法数字')
    if not vec:
        sys.exit(f'文件 {path} 为空或无有效数据')
    return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """计算余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        sys.exit('有向量范数为 0，无法计算余弦相似度')
    return dot / (norm1 * norm2)


def adjust_length(v1: List[float], v2: List[float], mode: str):
    """根据 mode 处理长度不一致的情况"""
    if len(v1) == len(v2):
        return v1, v2
    if mode == 'strict':
        sys.exit(f'向量长度不一致: {len(v1)} vs {len(v2)}')
    if mode == 'cut':
        n = min(len(v1), len(v2))
        return v1[:n], v2[:n]
    if mode == 'pad':
        diff = len(v1) - len(v2)
        if diff > 0:
            v2.extend([0.0] * diff)
        else:
            v1.extend([0.0] * (-diff))
        return v1, v2
    raise ValueError('未知长度处理模式')


def plot_vectors(v1, v2, name1, name2, cosim, output_path=None):
    """把两条曲线画在一张图上，并显示/保存"""
    plt.figure(figsize=(10, 4))
    x = range(len(v1))
    plt.plot(x, v1, label=name1, lw=1)
    plt.plot(x, v2, label=name2, lw=1)
    plt.title(f'Cosine similarity = {cosim:.6f}')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f'Figure saved to {output_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compute cosine similarity and plot two txt vectors.')
    parser.add_argument('--file1', default='/home/manu/tmp/saver_opencv.txt',
                        help='第一个向量文件路径')
    parser.add_argument('--file2', default='/home/manu/tmp/saver_py.txt',
                        help='第二个向量文件路径')
    parser.add_argument('-t', '--trim', choices=['strict', 'cut', 'pad'], default='strict',
                        help='长度不一致时的处理方式: strict=报错  cut=截断  pad=补0')
    parser.add_argument('-o', '--output', help='输出图像文件路径(如未指定则直接显示)')
    args = parser.parse_args()

    v1 = load_vector(args.file1)
    v2 = load_vector(args.file2)
    v1, v2 = adjust_length(v1, v2, args.trim)

    cosim = cosine_similarity(v1, v2)
    print(f'Cosine similarity = {cosim:.12f}')

    plot_vectors(
        v1,
        v2,
        os.path.basename(args.file1),
        os.path.basename(args.file2),
        cosim,
        args.output
    )


if __name__ == '__main__':
    main()
