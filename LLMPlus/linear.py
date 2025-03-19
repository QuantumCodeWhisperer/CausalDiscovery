import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import igraph 
import argparse
from 魔搭 import LLM
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import * 

def is_dag(W):
    G = igraph.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def notears_linear(X, lambda1, loss_type, model, max_iter=1000000, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        model(LLM): LLM
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape 
    model.causal_discovery()
    # double w_est into (w_pos, w_neg)
    matrix =  model.get_causal_matrix()
    w_est = np.concatenate([np.maximum(matrix, 0).flatten(), np.maximum(-matrix, 0).flatten()])
    rho, alpha, h =  1.0, 0.0, np.inf  
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    model.optimize(W_est)
    W_est = model.get_causal_matrix()
    return W_est

def parse_args():
    parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
    parser.add_argument('--X_path', type=str,default='dataSet/lucas.csv',help='n by p data matrix in csv format')
    parser.add_argument('--lambda1', type=float, default=0.00001, help='L1 regularization parameter')
    parser.add_argument('--loss_type', type=str, default='l2', help='l2, logistic, poisson loss')
    parser.add_argument('--W_path', type=str, default=current_dir+'//W_est.csv', help='p by p weighted adjacency matrix of estimated DAG in csv format')
    args = parser.parse_args()
    return args
   
if __name__ == '__main__':
    args = parse_args()
    model = LLM()
    model(args.X_path)
    data = pd.read_csv(args.X_path)
    X = data.values
    W_est = notears_linear(X, lambda1=args.lambda1, loss_type=args.loss_type,model=model)
    # assert is_dag(W_est)
    np.savetxt(args.W_path, W_est, delimiter=',', fmt='%.3f')
    display_graph(W_est,model.labels,model.path)