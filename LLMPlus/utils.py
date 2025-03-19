from graphviz import Digraph
import pandas as pd
import numpy as np
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def display_graph(adjMatrix,labels,path="causalGraph",withWeight=False):
    # 创建有向图对象
    dot = Digraph()
    # 设置图形的属性
    dot.attr(rankdir='TB', size='8,8')
    # 添加节点
    for label in labels:
        dot.node(label)
    # 添加边
    n,d = adjMatrix.shape
    assert n == d == len(labels)
    if withWeight:
        for i in range(n):
            for j in range(n):
                if adjMatrix[i][j]!= 0:
                    weight = str(adjMatrix[i][j])[:5]
                    dot.edge(labels[i], labels[j], label=weight)
    else:
        for i in range(n):
            for j in range(n):
                if adjMatrix[i][j] != 0:
                    weight = str(adjMatrix[i][j])[:5]
                    dot.edge(labels[i], labels[j])
    # 渲染图形并保存为 PNG 文件
    dot.render(path, view=True, format='svg')

def calculate_accuracy(B_true, B_est):
    # 检查输入矩阵是否为二维且形状相同
    if B_true.ndim != 2 or B_est.ndim != 2 or B_true.shape != B_est.shape:
        raise ValueError("输入矩阵必须是二维且形状相同。")
    d = B_true.shape[0]
    # 统计估计图矩阵B_est中非零元素的数量，即预测的边数
    nnz = np.count_nonzero(B_est)
    # 统计真实图矩阵B_true中非零元素的数量
    cond = np.count_nonzero(B_true)
    # 计算真正例（TP）
    true_pos = np.count_nonzero(np.logical_and(B_true != 0, B_est != 0))
    # 计算假正例（FP）
    false_pos = np.count_nonzero(np.logical_and(B_true == 0, B_est != 0))
    # 计算条件负例数
    cond_neg = d*d - cond
    # 计算错误发现率（FDR）
    fdr = false_pos / max(nnz, 1)
    # 计算真正例率（TPR）
    tpr = true_pos / max(cond, 1)
    # 计算假正例率（FPR）
    fpr = false_pos / max(cond_neg, 1)
    # 计算结构汉明距离（SHD）
    shd = np.count_nonzero(B_true != B_est)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': nnz}



