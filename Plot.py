# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:24:52 2023

@author: Tim Schell
"""
import pandas as pd
import matplotlib.pyplot as plt
"""
data = pd.read_excel('Validation.xlsx', index_col=0)
plt.figure(dpi=300)
plt.plot(data["Pyworld3-03"], color = "green", alpha = 1)
plt.plot(data["Stella"], color =  "red", alpha = 0.5)
plt.plot(data["LtG CD"], color = "blue" ,alpha = 0.5)
plt.title("comparison ecological footprint")
plt.xlabel('time [years]')
plt.ylabel('ef')
plt.grid(axis='x')
plt.grid(axis='y')
plt.xlim([1900, 2100])
plt.legend(['STELLA', 'PyWorld3-03', 'LtG CD'])
plt.show()
"""

"""
data = pd.read_excel('NRMSD graph from Recalibration23.xlsx', index_col=0)
data.plot(grid = True,  title = "NRMSD graph from Recalibration23", legend = False)
plt.figure(dpi=300)
plt.plot(data)
plt.title("NRMSD graph from Recalibration23")
plt.xlabel('Iteration')
plt.ylabel('NRMSD')
plt.grid(axis='x')
plt.grid(axis='y')
plt.savefig('NRMSD graph from Recalibration23.eps')
plt.show()
"""

data = pd.read_excel('LIBOR.xlsx', sheet_name = "1987-2023", index_col=0)
data.plot(grid = True,  title = "LIBOR index 1986 - 2023", legend = False)
plt.figure(dpi=300)
plt.plot(data)
plt.title("LIBOR index 1986 - 2023")
plt.xlabel('Years')
plt.ylabel('Percent')
plt.grid(axis='x')
plt.grid(axis='y')
#plt.savefig('NRMSD graph from Recalibration23.eps')
plt.show()