from uncertainties import unumpy
from uncertainties import ufloat
import numpy as np
import pandas as pd




#Calibrations
lcVoltage = 0.001 #mV
lcTime = 1 #s

lcHeaterVolateg = 0.01 #V
lcHeaterCurrent = 0.01 #A

zeroErrorVoltage = 0.002; #mV 





#Settings
heaterVoltage = 5.0 #V
heaterCurrent = 0.53 #A

timePeriod = ufloat(600, lcTime) #600.0 seconds
readingsInterval =   ufloat(10, lcTime) #10 seconds
distanceBetweenRods = ufloat(0.03, 0.001) #m








print('Reading file...')
df = pd.read_excel("thermal_diffusivity_brass/data.xlsx", skiprows=4)
print('File read successfully\n\n')



timeTC1 = df['Time TC1(s)']
timeTC2 = df['Time TC2(s)']
voltageTC1 = df['TC1(mV)']
voltageTC2 = df['TC2(mV)']



timeDataTC1 = unumpy.uarray(timeTC1, lcTime)
timeDataTC2 = unumpy.uarray(timeTC2, lcTime)
voltageDataTC1 = unumpy.uarray(voltageTC1.to_numpy()/1000-zeroErrorVoltage, lcVoltage)
voltageDataTC2 = unumpy.uarray(voltageTC2.to_numpy()/1000-zeroErrorVoltage, lcVoltage)




#Real Shit

tc1Cos = unumpy.cos(2*np.pi*timeDataTC1/timePeriod) * voltageDataTC1
tc1Sin = unumpy.sin(2*np.pi*timeDataTC1/timePeriod) * voltageDataTC1

tc2Cos = unumpy.cos(2*np.pi*timeDataTC2/timePeriod) * voltageDataTC2
tc2Sin = unumpy.sin(2*np.pi*timeDataTC2/timePeriod) * voltageDataTC2


sumTc1Cos = np.sum(tc1Cos) - 0.5 * (tc1Cos[0]+tc1Cos[-1])
sumTc1Sin = np.sum(tc1Sin) - 0.5 * (tc1Sin[0]+tc1Sin[-1])

sumTc2Cos = np.sum(tc2Cos) - 0.5 * (tc2Cos[0]+tc2Cos[-1])
sumTc2Sin = np.sum(tc2Sin) - 0.5 * (tc2Sin[0]+tc2Sin[-1])



I1Cos = sumTc1Cos*readingsInterval/timePeriod
I1Sin = sumTc1Sin*readingsInterval/timePeriod

I2Cos = sumTc2Cos*readingsInterval/timePeriod
I2Sin = sumTc2Sin*readingsInterval/timePeriod


Amp1 = unumpy.sqrt(np.square(I1Cos) +np.square(I1Sin)) 
Amp2 = unumpy.sqrt(np.square(I2Cos) +np.square(I2Sin)) 

phase1 = unumpy.arctan2(I1Sin, I1Cos) + np.pi
phase2 = unumpy.arctan2(I2Sin, I2Cos) + np.pi


alpha = -1*unumpy.log(Amp2/Amp1)/distanceBetweenRods
beta = (phase1 - phase2)/(-1*distanceBetweenRods)

D = (2*np.pi/timePeriod)/(2*alpha*beta)


print("Sum TC1 Cos", sumTc1Cos)
print("Sum TC1 Sin", sumTc1Sin)
print("Sum TC2 Cos", sumTc2Cos)
print("Sum TC2 Sin", sumTc2Sin)
print("I1 Cos", I1Cos)
print("I1 Sin", I1Sin)
print("I2 Cos", I2Cos)
print("I2 Sin", I2Sin)
print("Amp1", Amp1)
print("Amp2", Amp2)  
print("Phase1", phase1)
print("Phase2", phase2)
print("Alpha", alpha)
print("Beta", beta)
print("Diffusivity",D*1e6,"mm^2/s = ",D*1e4,"cm^2/s = ",D,"m^2/s")



df = pd.DataFrame(
    {
        "Time TC1": timeDataTC1,
        "Voltage TC1": voltageDataTC1,
        "TC1 Cos": tc1Cos,
        "TC1 Sin": tc1Sin,
        "Time TC2": timeDataTC2,
        "Voltage TC2": voltageDataTC2,
        "TC2 Cos": tc2Cos,
        "TC2 Sin": tc2Sin,
    }
)

df.to_csv("thermal_diffusivity_brass/output.csv", index=False)
print('\n\nSucessfully saved output to output.csv')