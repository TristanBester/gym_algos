import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def get_LS_fit(aves):
	lin_reg = LinearRegression()

	xs = np.array(range(len(aves)))
	aves = np.array(aves)

	lin_reg.fit(xs.reshape(-1,1), aves.reshape(-1,1))

	linear_fit = lin_reg.predict(xs.reshape(-1,1))
	return linear_fit

def moving_ave(ls, n=10):
	moving_aves = []
	for i in range(n,len(ls), n):
		val = np.sum(ls[i-n:i])
		moving_aves.append(val/n)
	return moving_aves

def create_surface_plot(x, y, z, x_label, y_label, title):
    '''Create a surface plot using the given data.'''
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, z, cmap='jet', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def create_contour_plot(x, y, z, levels, labels,x_label, y_label, title):
	'''Create contour plot using the given data.'''
	cs = plt.contourf(x,y,z, levels=levels, cmap='cool')
	proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
    		 for pc in cs.collections]
	plt.legend(proxy, labels)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.show()
