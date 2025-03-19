from openai import OpenAI
import pandas as pd
import numpy as np
import sys
import os
current_dir = os.path.dirname(__file__)

class LLM:
    """
    本模型用于因果发现。
    """
    def __init__(self,key="dc8df87a-25c9-40eb-8059-b9e0627c6a63",
                 url="https://api-inference.modelscope.cn/v1/",
                 model="deepseek-ai/DeepSeek-R1"):
        self.client = OpenAI(api_key=key, base_url=url)
        self.model = model
        self.answer = ""
        self.path = ""
        self.labels = []
        self.data = None
       
    def __call__(self,file_path="dataSet/lucas.csv"):
        data = pd.read_csv(file_path)
        self.path = current_dir+"//"+file_path.split('/')[-1].split('.')[0]+'_est_graph'
        self.labels = data.columns.tolist()
        self.data = data.values

    def data_to_string(self,labels,data):
        if self.path == "":
            self.__call__()
        data = ' '.join(labels)+'\n'
        data += '\n'.join(' '.join(map(str, row)) for row in data)
        return data
    
    def set_answer(self,response):
        answer = ""
        # 遍历流式响应的每个部分
        for chunk in response:
            # 检查当前部分是否包含内容
            if chunk.choices[0].delta.content is not None:
                # 将当前部分的内容添加到最终结果中
                answer += chunk.choices[0].delta.content
        self.answer = answer

    def causal_discovery(self):
        data = self.data_to_string(self.labels,self.data)
        messages = [{"role": "system", "content": "你是用于因果发现的人工智能助手."},
                    {"role": "user", "content": "我会给你一份数据,请你进行深度思考."},
                    {"role": "user", "content": data},
                    {"role": "user", "content": "根据以上数据返回一个因果发现矩阵(有向无环图).除此之外不要返回任何东西,也不需要注释."}]
        response = self.client.chat.completions.create(model=self.model,messages=messages,stream=True)
        self.set_answer(response)

    def optimize(self,adjMatrix):
        data = self.data_to_string(self.labels,self.data)
        matrix = self.data_to_string(self.labels,adjMatrix)
        messages = [{"role": "system", "content": "你是用于因果发现的人工智能助手."},
                    {"role": "user", "content": "我会给你一份数据和一个因果发现矩阵(表示一个有向无环图),请你进行深度思考,根据该数据和变量名对该因果发现矩阵优化,使其更具可解释性."},
                    {"role": "user", "content": "数据:"},
                    {"role": "user", "content":data},
                    {"role": "user", "content": "因果发现矩阵:"},
                    {"role": "user", "content":matrix},
                    {"role": "user", "content": "返回优化后的因果发现矩阵(表示一个有向无环图).除此之外不要返回任何东西,也不需要注释."}]
        response = self.client.chat.completions.create(model=self.model,messages=messages,stream=True)
        self.set_answer(response)

    def get_causal_matrix(self):
        if self.answer == "":
            return []
        n = self.data.shape[1]
        causal_matrix = np.zeros((n, n))
        digits = (int(ch) for ch in self.answer if ch.isdigit())
        for i in range(n):
            for j in range(n):
                causal_matrix[i][j] = next(digits)
        return causal_matrix

