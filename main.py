import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)

class SuperDuperCoolAsFuckRegressorYoooooo:
    def __init__(self, dim, N, RandomInit=True, lr=0.001):
        self.lr = lr
        self.N = N
        self.w0 = 0.1
        self.lastterm1 = np.zeros(dim)
        self.lastterm2 = np.zeros((dim, dim))

        if RandomInit:
            self.wi = 0.5-np.random.rand(dim)
            self.wij = 0.5-np.random.random((dim, dim))
        else:
            self.wi = 0.5-np.ones(dim)
            self.wij = 0.5-np.ones((dim, dim))
        print('wi: ', self.wi)
        print('wij: ', self.wij)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_d(self, x):
        return x * (1-x)
    
    def forward(self, x):
        self.lastterm1 = x
        self.lastterm2 = np.outer(x, x)
        x = self.w0 + np.dot(self.wi, self.lastterm1.T) + np.sum(np.matmul(self.wij, self.lastterm2))
        return self.sigmoid(x)
    
    def calcloss(self, pred, y):
        error = (pred - y)
        loss = error*error*0.5
        return loss/self.N, error
    
    def step(self, error, pred):
        self.w0 -= self.lr * error * self.sigmoid_d(pred)
        self.wi -= self.lr * error * self.sigmoid_d(pred) * self.lastterm1
        self.wij -= self.lr* error * self.sigmoid_d(pred) * self.lastterm2


# load X
data_x = pd.read_csv("data_X.csv")
data_X = []
for x in zip(data_x['GRE_score'], data_x['TOFEL_score'], data_x['University_rating'], data_x['SOP'], data_x['LOR'], data_x['CGPA'], data_x['Research']):
    data_X.append(x)

# normalize X
data_X = np.asarray(data_X)
data_X /= data_X.max(axis=0)
print('X samples: ', data_X[:3])

# load Y
data_y = pd.read_csv("data_T.csv")
data_Y = []
for y in data_y['Chance_of_Admit']:
    data_Y.append(y)
print('Y samples: ', data_Y[:3])

# initialize model
N = data_X.shape[0]
model = SuperDuperCoolAsFuckRegressorYoooooo(dim=data_X.shape[1], N=N, lr=0.001)
preds = []
losses = []
errors = []

epochs = 1000
# fit
# For convenience of testing, I mixed procedural style and OOP style. You should wrap the inner loop into the class object
for e in range(0, epochs):
    loss_epoch = []
    error_epoch = []

    for idx in range(0, N):
        pred = model.forward(data_X[idx])
        preds.append(pred)
        loss, error = model.calcloss(pred, data_Y[idx])
        model.step(error, pred)

        loss_epoch.append(loss)
        error_epoch.append(error)
    
    losses.append(np.sum(np.asarray(loss_epoch))) 
    error = np.sum(np.asarray(error_epoch)/N)
    errors.append(error)
        

# plot the loss and error curve
fig = plt.figure()
plt.plot(losses)
fig.suptitle('Loss every epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.close('all')

fig = plt.figure()
plt.plot(errors)
fig.suptitle('Error every epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
plt.clf()
plt.close('all')