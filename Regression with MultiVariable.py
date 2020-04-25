#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  


# In[88]:


path="E:\\Machine_learning_projects\\test\\data2.csv"
mydata=pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])


# In[89]:


mydata.head()


# In[90]:


mydata.describe()


# In[91]:


#rescaling data 
mydata=(mydata-mydata.mean())/mydata.std()
mydata.head(10)


# In[92]:


#add ones columns
mydata.insert(0,'Ones',1)


# In[93]:


mydata.head()


# In[94]:


#seperate x(training data) from y(target variable)
cols=mydata.shape[1]
X=mydata.iloc[:,:cols-1]
y=mydata.iloc[:,cols-1:cols]


# In[95]:


print(X.shape)
print(X.head())


# In[96]:


print(y.shape)
print(y.head())


# In[97]:


#convert to matrix and initialize theta
X=np.matrix(X)
y=np.matrix(y)
theta=np.matrix(np.array([0,0,0]))


# In[98]:


X.shape


# In[99]:


#initialize the data for alpha(learning rate) and iterations
alpha=0.1
iters=100


# In[100]:


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost


# In[101]:


def computeCost(X,y,theta):
    z=np.power(((X*theta.T)-y),2)
    print('z \n',z)
    print('m' ,len(X))
    return np.sum(z)/(2*len(X))
#print('computeCost(X,y,theta)=',computeCost(X,y,theta))


# In[102]:


g2,cost=gradientDescent(X, y, theta, alpha, iters)


# In[103]:


g2


# In[104]:


cost


# In[111]:


final_cost=cost.min()
print('the min cost =\n with theta = ',final_cost,g2)


# In[114]:


# get best fit line for Size vs. Price

x = np.linspace(mydata.Size.min(), mydata.Size.max(), 100)
print('x \n',x)
print('g \n',g2)

f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)

# draw the line for Size vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(mydata.Size, mydata.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# In[115]:



# get best fit line for Bedrooms vs. Price

x = np.linspace(mydata.Bedrooms.min(), mydata.Bedrooms.max(), 100)
print('x \n',x)
print('g \n',g2)

f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(mydata.Bedrooms,mydata.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Bedrooms vs. Price')


# In[117]:


# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# In[ ]:




