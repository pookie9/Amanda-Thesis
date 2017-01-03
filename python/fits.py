import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def show_model(X,y,model):
    print X.shape,y.shape
    model=model()
    model.fit(X,y)
    plot_x=np.array([model.predict(y_val)-x_val for x_val,y_val in zip(X,y)])
    plt.scatter(plot_x,y)
    plt.show()
#data_file=open("in vitro VE822 drug levels .csv")
data=np.genfromtxt("in vitro VE822 drug levels .csv",delimiter=',')[1:,:2]
X=data[:,0]
X=np.reshape(X,(X.shape[0],1))
y=np.log(data[:,1])

show_model(X,y,LinearRegression)
