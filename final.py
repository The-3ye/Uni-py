"""

2024 dec 03 
final // intro prog ingenieurs UdeS
Raphaelle Fortin

Pratique a l'examen final                                 

"""

#---------///---------///---------///---------

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import linregress 

#---------///---------///---------///---------
#~~~# EXAMPLE 1 #~~~#

"""
#optimisation

 #EQ functions for optimisation
def funct1(param):   
    x = param[0]
    f = (x**2)-(4*x)+5 
    return f

def funct2(param):   
    x = param[0]
    f = -(-x**2 + 4*x -1)
    return f

def funct3(param):   
    x = param[0]
    f = (e**x)+(x**2)-(3*x)
    return f
    
def funct4(param):   
    x = param[0]
    f = -(-np.exp(-x+2)-x)
    return f

def funct5(param):   
    x = param[0]
    f = (e**(-x))+x
    return f

def funct6(param):   
    x = param[0]
    f = -(np.exp(x) - 3*x**2)
    return f

#initial variables 
e = np.e
init0 = [0.0,0.0] #x ; y
init2 = [2.0,0.0] #x ; y

#optimisation code
result1 = minimize(funct1,init2)
x1 = result1.x[0] # .x gives optimisation solution
min1 = result1.fun # .fun gives value of function
print(f"function 1 minimum is at {x1} with value {min1}")

result2 = minimize(funct2,init0)
x2 = result2.x[0]
max2 = result2.fun
print(f"function 2 maximum is at {x2} with value {max2}")

result3 = minimize(funct3,init0)
x3 = result3.x[0]
min3 = result3.fun
print(f"function 3 minimum is at {x3} with value {min3}")

result4 = minimize(funct4,init2)
x4 = result4.x[0]
max4 = result4.fun
print(f"function 4 maximum is at {x4} with value {max4}")

result5 = minimize(funct5,init0)
x5 = result5.x[0]
min5 = result5.fun
print(f"function 5 minimum is at {x5} with value {min5}")

result6 = minimize(funct6,init0)
x6 = result6.x[0]
max6 = result6.fun
print(f"function 6 maximum is at {x6} with value {max6}")

#graph code

 #plot functions
def f1(x):
    return (x**2)-(4*x)+5

def f2(x):
    return -x**2 + 4*x -1

def f3(x):
    return (e**x)+(x**2)-(3*x)

def f4(x):
    return (-(e)**(-x+2))-x

def f5(x):
    return (e**(-x))+x

def f6(x):
    return np.exp(x) - 3*x**2

 #plot values
xValues = np.linspace(-20,20, 200)
x6Values = np.linspace(-2,3, 200)

y1Values = f1(xValues)
y2Values = f2(xValues)
y3Values = f3(xValues)
y4Values = f4(xValues)
y5Values = f5(xValues)
y6Values = f6(xValues)

 #start plotting
fig, ax = plt.subplots(2, 3) #2 lines, 3 columns

ax[0,0].set_title('function 1') #title
ax[0,0].plot(xValues,y1Values, color='#b16286') #curve 
ax[0,0].scatter(x1,min1, color='#d65d0e') #extremum

ax[0,1].set_title('function 2')
ax[0,1].plot(xValues,y2Values, color='#b16286')
ax[0,1].scatter(x2,max2, color='#d65d0e')

ax[0,2].set_title('function 3')
ax[0,2].plot(xValues,y3Values, color='#b16286')
ax[0,2].scatter(x3,min3, color='#d65d0e')

ax[1,0].set_title('function 4')
ax[1,0].plot(xValues,y4Values, color='#b16286')
ax[1,0].scatter(x4,max4, color='#d65d0e')

ax[1,1].set_title('function 5')
ax[1,1].plot(xValues,y5Values, color='#b16286')
ax[1,1].scatter(x5,min5, color='#d65d0e')

ax[1,2].set_title('function 6')
ax[1,2].plot(x6Values,y6Values, color='#b16286')
ax[1,2].scatter(x6,max6, color='#d65d0e')

plt.show()
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 2 #~~~#

"""
df = pd.read_excel('exemple2.xlsx')

slope, intercept, r, p, se = linregress(df['x'], df['y'])
yRegress = (df['x'] * slope) + intercept


print(df.head(5))
print(f"slope (a) is : {slope}")
print(f"intercept (b) is : {intercept}")
print(f"R squared is : {r**2}")


plt.scatter(df['x'],df['y'], color='#b16286', marker='^')
plt.plot(df['x'],yRegress, color='#d79921', ls='--')
plt.show()
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 3 #~~~#

"""
xMaxVal = 6
xMinVal = -2
xAmount = (xMaxVal - xMinVal) / 0.1

def f(x):
    return (0.9*(x**2))-(4*x)+5


xValues = np.linspace(xMinVal,xMaxVal, int(xAmount))
yValues = f(xValues)

df = pd.DataFrame({'x': xValues, 'y': yValues})
df.to_excel('exemple3.xlsx')

result = minimize(f,[0.0])
xMin = result.x[0]
yMin = result.fun

slope, intercept, r, p, se = linregress(xValues,yValues)
yRegress = (xValues*slope)+intercept
rSquared = f"R² = {r**2:.4f}" #f string is used to make it a label on graph

plt.plot(xValues,yValues, color='#cc241d', label='EQ : (0.9x^2)-4x+5')
plt.plot(xValues,yRegress, color='#fb4934', ls='--', label=rSquared)
plt.scatter(xMin,yMin, color='#fabd2f')

plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.show()
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 4 #~~~#

"""
xMinValue = 0.0
xMaxValue = 6.0
xAmount = (xMinValue+xMaxValue)/0.1

def f(x):
    return (-(x**2))+(6*x)-9+np.exp(-x)

def minusf(x):
    return -((-(x**2))+(6*x)-9+np.exp(-x))

xValues = np.linspace(xMinValue,xMaxValue, int(xAmount))
yValues = f(xValues)

df= pd.DataFrame({'x':xValues,'y':yValues})
df.to_excel('exemple4.xlsx')

result = minimize(minusf,[0.0])
xMax = result.x[0]
yMax = result.fun

slope, intercept, r, p, se = linregress(xValues,yValues)
rSquared = f"R²= {r**2:.4f}"
yRegress = (xValues*slope)+intercept

plt.plot(xValues,yValues, color='#98971a')
plt.plot(xValues,yRegress, color='#b8bb26', ls=':', label=rSquared)
plt.scatter(xMax,yMax, color='#d3869b')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 5 #~~~#


#---------///---------///---------///---------
#~~~# EXAMPLE 6 #~~~#

 #data
df = pd.read_excel('bacterial_growth_data.xlsx')
t = df['time (h)'].values
logN = df['log(N)'].values

 #linear regreassion
slope, intercept, r, p, se = linregress(t,logN)

 # adjusted parameters 
n0_fitted = np.exp(intercept)

xTime = np.linspace(0, 10, 100)
yLogRegress = (slope * xTime) + intercept
n = np.exp(yLogRegress)

print(f"Inital Amount of Bacteria : {n0_fitted:.4f}")
print(f"Growth Rate : {slope:.4f}")
print(f"R² : {r**2:.4f}")
 #plotting
plt.scatter(t,np.exp(logN), color='#dfaf87', label='Experimental Data')
plt.plot(xTime,n, color='#83a598', ls=':', label='Adjusted Model')

plt.title('Adjustement of Bacteria Growth Model')
plt.xlabel('Time (h)')
plt.ylabel('Amount of bacteria (n)')
plt.legend()
plt.show()





