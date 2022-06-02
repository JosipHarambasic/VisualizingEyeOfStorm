import numpy as np
import matplotlib.pyplot as plt

dim= 10
xarray= np.arange(-dim,dim)
yarray= np.arange(-dim,dim)

x,y = np.meshgrid(xarray,yarray)

print(-x,y)