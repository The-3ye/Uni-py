"""

2024 dec 03 
final // intro prog ingenieurs UdeS
                  

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

#inneficient way to do it, online solution is better

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
 #innefficient part, could be merged with above functions
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


plt.scatter(df['x'],df['y'], color='#b16286', marker='^') #color and marker are cosmetic
plt.plot(df['x'],yRegress, color='#d79921', ls='--') #ls -> linestyle, cosmetic
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

plt.plot(xValues,yValues, color='#cc241d', label='EQ : 0.9x²-4x+5')
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

"""
 #data
df = pd.read_excel('bacterial_growth_data.xlsx')
t = df['time (h)'].values
logN = df['log(N)'].values

 #linear regreassion
slope, intercept, r, p, se = linregress(t,logN)

xTime = np.linspace(0, 10, 100)
yLogRegress = (slope * xTime) + intercept

 # adjusted parameters 
n0_fitted = np.exp(intercept)
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
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 7 #~~~#

"""
 #values
df = pd.read_excel('fluid_flow_data.xlsx')
q = df.iloc[:,0].values
deltaP = df.iloc[:,1].values

 #logs to work with EQ
ln_q = np.log(q)
ln_deltaP = np.log(deltaP)

 #linear regression
slope, intercept, r, idk, se = linregress(ln_q,ln_deltaP)

k = np.exp(intercept)

 #fitted curve values
ln_dP_fit = (slope*ln_q)+intercept
deltaP_fit = np.exp(ln_dP_fit)

 #begin plotting
plt.scatter(q,deltaP, color='#d79921', label='Experimental Data')
plt.plot(q, deltaP_fit, color='#fb4934', ls=':', label="Adjusted Model")

plt.text(0.25,150, f"k = {k:.2f}") #not learned in class, alternative to printing k in console
plt.title('Adjustement of Darcy-Weisbach Equation')
plt.xlabel('Q, m³/s')
plt.ylabel('ΔP, Pa')
plt.legend() plt.show()
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 8 #~~~#

"""
 #Equation
def f(params):
    a = params[0]
    b = params[1]
    v1 = [a-2,b+3,-1,1]
    v2 = [a-2,b+3,1,1]
    scalar = np.dot(v1,v2)
    return scalar**2

 #optimal parameters
result = minimize(f,[0.0,0.0])
aOpt = result.x[0]
bOpt = result.x[1]
minVal = result.fun

aOptimal = round(aOpt, 2)
bOptimal = round(bOpt, 2)
minValue = round(minVal, 2)

 #send to text file
with open('exemple8.txt', 'w') as f:
    f.write(f"alpha = {aOptimal}\n")
    f.write(f"beta = {bOptimal}\n")
    f.write(f"function minimum = {minValue}")
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 9 #~~~#

"""
 # Equation
def f(parameters):
    a = parameters[0]
    b = parameters[1]
    v1 = [a+2, b-5, -3, 3]
    v2 = [a+2, b-5, 1, 1]
    return (np.dot(v1,v2))**2

 #optimal  parameters
result = minimize(f,[0,0])
aOptimal = round(result.x[0], 2)
bOptimal = round(result.x[1], 2)
minValue = round(result.fun, 2)

 #shows results, to send to text file, see EXAMPLE 8
print(f"Optimal a value is : {aOptimal}")
print(f"Optimal b value is : {bOptimal}")
print(f"Minimal function value is : {minValue}")
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 10 #~~~#

"""
 #Equation
def f(parameters):
    a = parameters[0]
    b = parameters[1]
    v1 = [a+2, b+5, 10, 1]
    v2 = [a-2, b+3, 1, 1]
    return (np.dot(v1,v2)**2)

 #optimal  parameters
result = minimize(f,[0,0])
aOptimal = round(result.x[0], 2)
bOptimal = round(result.x[1], 2)
minValue = round(result.fun, 2)

 #shows results, to send to text file, see EXAMPLE 8
print(f"Optimal a value is : {aOptimal}")
print(f"Optimal b value is : {bOptimal}")
print(f"Minimal function value is : {minValue}")
"""

#---------///---------///---------///---------
#~~~# EXAMPLE 11 #~~~#

"""
 #Equation
def f(parameters):
    a = parameters[0]
    b = parameters[1]
    return ((a**2)+(b**2)+(4*b)+10)**2

 #optimal parameters
result = minimize(f,[0,0])
aOptimal = round(result.x[0], 2)
bOptimal = round(result.x[1], 2)
minValue = round(result.fun, 2)

 #shows results, to send to text file, see EXAMPLE 8
print(f"Optimal a value is : {aOptimal}")
print(f"Optimal b value is : {bOptimal}")
print(f"Minimal function value is : {minValue}")

print("-----~~~~~~~-----")

#does not give different values when changing initial guess as expected
#>

 #TESTEZ-VOUS Equation
def g(parameters):
    a = parameters[0]
    b = parameters[1]
    return ((a**2)-(b**2)+(4*b)+10)**2

 #optimal parameters
result2 = minimize(f,[0.5,-10]) #<~~!~~ changed parameters have no effect
aOptimal2 = round(result2.x[0], 2)
bOptimal2 = round(result2.x[1], 2)
minValue2 = round(result2.fun, 2)

 #shows results, to send to text file, see EXAMPLE 8
print(f"Optimal a value is : {aOptimal2}")
print(f"Optimal b value is : {bOptimal2}")
print(f"Minimal function value is : {minValue2}")
#>
#>


"""

#---------///---------///---------///---------
#~~~# EXAMPLE 12 #~~~#
 #VBA bruh

#---------///---------///---------///---------
#~~~# EXAMPLE 13 #~~~#
 #VBA bruh

#---------///---------///---------///---------
#~~~# EXAMPLE 14 #~~~#


"""
 #DataFrame creation
scoresDF = pd.read_excel('student_scores.xlsx')


 #group mean for every subject
 # subject -> DataFame column 
mathMean = np.mean( scoresDF['math_score'])
physMean = np.mean( scoresDF['physics_score'])
chemMean = np.mean( scoresDF['chemistry_score'])
generalMean = np.mean( scoresDF['total_score'])

bestStudentValue = np.max(scoresDF['total_score'])  #score of best student

bestStudentEntry = scoresDF['total_score'].idxmax() #finds ID (row) of max val for condition

bestStudentRow = scoresDF.loc[bestStudentEntry] #.loc selects row based on condition ^^
bestStudentID = bestStudentRow['student_id']    # we want seleccted row in column of choice, here 'st_id'

 #better than some number (here 15 / 60)
percentAbove15 = (scoresDF['total_score'] >= 15).mean() * 100 #yes, here 100% of the class got above 15

 #show results
print(f"Class mean score for Math is : {mathMean}/20")
print(f"Class mean score for Physics is : {physMean}/20")
print(f"Class mean score for Chemistry is : {chemMean}/20")
print(f"\nClass mean total score is : {generalMean}/60")
print(f"{percentAbove15} % of the class got higher than 15\n")
print("-----~~~~~~~-----\n")
print(f"Best student is : {bestStudentID} with score : {bestStudentValue}/60")

"""

#---------///---------///---------///---------
#~~~# EXAMPLE 15 #~~~#






