import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)
import time

# load X
data_x = pd.read_csv("data_X.csv")
data_X = []
for x in zip(data_x['GRE_score'], data_x['TOFEL_score'], data_x['University_rating'], data_x['SOP'], data_x['LOR'], data_x['CGPA'], data_x['Research']):
    data_X.append(x)

# normalize X
data_X = np.asarray(data_X)
#data_X /= data_X.max(axis=0)
print('X samples: ', data_X[:3])

# load Y
data_y = pd.read_csv("data_T.csv")
data_Y = []
for y in data_y['Chance_of_Admit']:
    data_Y.append(y)
data_Y = np.asarray(data_Y)
print('Y samples: ', data_Y[:3])

class SuperDuperCoolAsFuckRegressorYoooooo:
    def __init__(self, dim, N, M=2, RandomInit=True, lb=0.1):
        self.lb = lb
        self.dim = dim
        self.N = N
        self.M = M

        asehgfbk = 0
        for i in range(0, M+1):
            asehgfbk += np.power(dim, i)
        self.w = np.ones(asehgfbk)

        print('w shape: ', self.w.shape)
        print('N: ', self.N)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_d(self, x):
        return x * (1-x)
    
    def get_psi(self, x):
        self.psi = []
        
        if self.M==2:
            x_row = np.ones(self.w.shape[0])
            for i in range(0, self.N):
                x_row[0] = 1.0
                x_row[1:self.dim+1] = self.sigmoid(x[i])
                x_row[self.dim+1:] = self.sigmoid(np.outer(x[i], x[i]).flatten())
                self.psi.append(x_row)

        else:
            x_row = np.ones(self.w.shape[0])
            for i in range(0, self.N):
                x_row[0] = 1.0
                x_row[1:self.dim+1] = self.sigmoid(x[i])
                start = self.dim+1
                for j in range(2, self.M+1):
                    pura = x[i]
                    for k in range(1, j):
                        pura = np.outer(pura, x[i])
                    x_row[start:start+np.power(self.dim, j)] = self.sigmoid(pura.flatten())
                    start += np.power(self.dim, j)
                    #print('start: ', start)
                self.psi.append(x_row)
        
        self.psi = np.asarray(self.psi)
    
    def calcloss(self, preds, y):
        error = np.subtract(preds, y)
        loss = np.sum(np.square(error)) / self.N
        print('RMS Loss: ', loss)
        return loss
    
    def train(self, y):
        print('Psi shape: ', self.psi.shape)
        p1 = np.dot(self.psi.T, self.psi)
        print('p1 done')
        p2 = np.linalg.pinv(p1+np.identity(p1.shape[0])*self.lb)
        print('p2 done')
        p3 = np.dot(p2, self.psi.T)
        print('p3 done')
        self.w = np.dot(p3, y)
        print('Trained w shape: ', self.w.shape)
    
    def predict(self, y):
        preds = []
        for i in range(0, self.N):
            preds.append(np.sum(np.dot(self.w, y[i])))
        preds = np.asarray(preds)
        #preds = np.sum(np.dot(predictor, self.psi.T), axis=1)

        return preds
    
    def eval(self, y):
        preds = self.predict(y)
        self.calcloss(preds, y)


M = 3
model = SuperDuperCoolAsFuckRegressorYoooooo(dim=data_X[0].shape[0], N=data_X.shape[0], M=M, lb=0.1)
start = time.time()
model.get_psi(data_X)
model.train(data_Y)
model.eval(data_Y)
end = time.time()
print('Elapsed {} seconds for M={}'.format(end - start, M))