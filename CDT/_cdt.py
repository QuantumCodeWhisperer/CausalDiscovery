import numpy as np
import pandas as pd
import networkx as nx
from cdt.causality.graph import PC,GES,CGNN
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import * 


class CDT():
    def __init__(self,file_path):
        self.root = current_dir+'//'+file_path.split('/')[-1].split('.')[0]
        self.path = ""
        self.data = pd.read_csv(file_path)
        self.labels = self.data.columns.tolist()
        self.method = None

    def __call__(self,method):
        self.method = method
        self.path = self.root+'_'+method.__class__.__name__+'_est_graph'

    def predict(self):
        graph = self.method.predict(self.data)
        B_est = nx.to_numpy_array(graph)
        return B_est

if __name__ == "__main__":
    file_path = "dataSet/lucas.csv"
    model = CDT(file_path)
    model(PC())
    B_est = model.predict()
    display_graph(B_est,model.labels,model.path)
    model(GES())
    B_est = model.predict()
    display_graph(B_est,model.labels,model.path)
    # model(CGNN())
    # B_est = model.predict()
    # display_graph(B_est,model.labels,model.path)
