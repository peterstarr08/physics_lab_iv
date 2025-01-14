import numpy as np
import autograd.numpy as autogradnp   # Thinly-wrapped version of Numpy
from autograd import grad
import pandas as pd

#Calibrations
lcVoltage = 0.001 #mV
lcTime = 1 #s

lcHeaterVolateg = 0.01 #V
lcHeaterCurrent = 0.01 #A

zeroErrorVoltage = 0.007; #mV 

#Settings
heaterVoltage = 5.0 #V
heaterCurrent = 0.53 #A


# Time in s
timeDataTC1 = np.array([1215, 1225, 1235, 1245, 1255, 1265, 1275, 1285, 1295, 1305, 1315, 1325, 1335, 1345, 1355, 1365, 1375, 1385, 1395, 1405, 1415, 1425, 1435, 1445, 1455, 1465, 1475, 1485, 1495, 1505, 1515, 1525, 1535, 1545, 1555, 1565, 1575, 1585, 1595, 1605, 1615, 1625, 1635, 1645, 1655, 1665, 1675, 1685, 1695, 1705, 1715, 1725, 1735, 1745, 1755, 1765, 1775, 1785, 1795
], dtype=np.float64)

# Normalized voltage in mV and calibrated
voltageDataTC1 = (np.array([20, 32, 42, 51, 57, 62, 66, 69, 71, 73, 74, 75, 76, 77, 77, 77, 77, 76, 76, 76, 75, 75, 74, 74, 74, 73, 73, 73, 73, 71, 60, 48, 39, 31, 25, 21, 17, 14, 12, 10, 9, 7, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6
], dtype=np.float64) )/1000 -zeroErrorVoltage

# Time in s
timeDataTC2 = np.array([2415, 2425, 2435, 2445, 2455, 2465, 2475, 2485, 2495, 2505, 2515, 2525, 2535, 2545, 2555, 2565, 2575, 2585, 2595, 2605, 2615, 2625, 2635, 2645, 2655, 2665, 2675, 2685, 2695, 2705, 2715, 2725, 2735, 2745, 2755, 2765, 2775, 2785, 2795, 2805, 2815, 2825, 2835, 2845, 2855, 2865, 2875, 2885, 2895, 2905, 2915, 2925, 2935, 2945, 2955, 2965, 2975, 2985, 2995
], dtype=np.float64)

# Nomrmalized volateg mV and calibrated
voltageDataTC2 =( np.array([134, 189, 233, 267, 291, 312, 329, 343, 355, 365, 374, 381, 389, 395, 400, 405, 409, 413, 417, 420, 424, 426, 429, 432, 434, 436, 438, 440, 442, 434, 378, 310, 268, 232, 208, 187, 171, 158, 145, 136, 128, 120, 113, 107, 102, 97, 93, 89, 86, 83, 80, 77, 74, 72, 70, 68, 66, 64, 62
], dtype=np.float64) )/1000 - zeroErrorVoltage 

timePeriod = 600.0 #seconds
readingsInterval = 10 #seconds
distanceBetweenRods = 0.03; #m

#Error propagation
def tcCos(timeTC, voltageTC, timePeriod):
    return autogradnp.cos(2*np.pi*timeTC/timePeriod) * voltageTC

def tcSin(timeTC, voltageTC, timePeriod):
    return autogradnp.sin(2*np.pi*timeTC/timePeriod) * voltageTC


gradientTcCos = [grad(tcCos, 0), grad(tcCos, 1), grad(tcCos, 2)]
gradientTcSin = [grad(tcSin, 0), grad(tcSin, 1), grad(tcSin, 2)]

def errorTcCos(timeTc, _timeTc, voltageTc, _voltageTc, timePeriod, _timePeriod):
    return np.sqrt(np.square(gradientTcCos[0](timeTc, voltageTc, timePeriod)*_timeTc) + np.square(gradientTcCos[1](timeTc, voltageTc, timePeriod)*_voltageTc) + np.square(gradientTcCos[2](timeTc, voltageTc, timePeriod)*_timePeriod))

def errorTcSin(timeTc, _timeTc, voltageTc, _voltageTc, timePeriod, _timePeriod):
    return np.sqrt(np.square(gradientTcSin[0](timeTc, voltageTc, timePeriod)*_timeTc) + np.square(gradientTcSin[1](timeTc, voltageTc, timePeriod)*_voltageTc) + np.square(gradientTcSin[2](timeTc, voltageTc, timePeriod)*_timePeriod))

numpyErrorTcCos = np.vectorize(errorTcCos)
numpyErrorTcSin = np.vectorize(errorTcSin)



def Amp(ICos, ISin):
    return autogradnp.sqrt(np.square(ICos) + np.square(ISin))

gradientAmp = [grad(Amp, 0), grad(Amp, 1)]

def errorAmp(ICos, _ICos, ISin, _ISin):
    return np.sqrt(np.square(gradientAmp[0](ICos, ISin)*_ICos) + np.square(gradientAmp[1](ICos, ISin)*_ISin))



def phase(ISin, ICos):
    return autogradnp.arctan2(ISin, ICos) + np.pi

gradientPhase = [grad(phase, 0), grad(phase, 1)]

def errorPhase(ISin, _ISin, ICos, _ICos):
    return np.sqrt(np.square(gradientPhase[0](ISin, ICos)*_ISin) + np.square(gradientPhase[1](ISin, ICos)*_ICos))



def alpha(Amp1, Amp2, distanceBetweenRods):
    return autogradnp.log(Amp2/Amp1)/distanceBetweenRods

gradientAlpha = [grad(alpha, 0), grad(alpha, 1), grad(alpha, 2)]

def errorAlpha(Amp1, _Amp1, Amp2, _Amp2, distanceBetweenRods, _distanceBetweenRods):
    return np.sqrt(np.square(gradientAlpha[0](Amp1, Amp2, distanceBetweenRods)*_Amp1) + np.square(gradientAlpha[1](Amp1, Amp2, distanceBetweenRods)*_Amp2) + np.square(gradientAlpha[2](Amp1, Amp2, distanceBetweenRods)*_distanceBetweenRods))


def beta(phase1, phase2, distanceBetweenRods):
    return (phase1 - phase2)/(-1*distanceBetweenRods)

gradientBeta = [grad(beta, 0), grad(beta, 1), grad(beta, 2)]

def errorBeta(phase1, _phase1, phase2, _phase2, distanceBetweenRods, _distanceBetweenRods):
    return np.sqrt(np.square(gradientBeta[0](phase1, phase2, distanceBetweenRods)*_phase1) + np.square(gradientBeta[1](phase1, phase2, distanceBetweenRods)*_phase2) + np.square(gradientBeta[2](phase1, phase2, distanceBetweenRods)*_distanceBetweenRods))


def D(timePeriod, alpha, beta):
    return (2*np.pi/timePeriod)/(2*alpha*beta)

gradientD = [grad(D, 0), grad(D, 1), grad(D, 2)]

def errorD(timePeriod, _timePeriod, alpha, _alpha, beta, _beta):
    return np.sqrt(np.square(gradientD[0](timePeriod, alpha, beta)*_timePeriod) + np.square(gradientD[1](timePeriod, alpha, beta)*_alpha) + np.square(gradientD[2](timePeriod, alpha, beta)*_beta))

#Real shit begins here


tc1Cos = np.cos(2*np.pi*timeDataTC1/timePeriod) * voltageDataTC1
_tc1Cos = numpyErrorTcCos(timeDataTC1, lcTime, voltageDataTC1, lcVoltage, timePeriod, lcTime)
tc1Sin = np.sin(2*np.pi*timeDataTC1/timePeriod) * voltageDataTC1
_tc1Sin = numpyErrorTcSin(timeDataTC1, lcTime, voltageDataTC1, lcVoltage, timePeriod, lcTime)

tc2Cos = np.cos(2*np.pi*timeDataTC2/timePeriod) * voltageDataTC2
_tc2Cos = numpyErrorTcCos(timeDataTC2, lcTime, voltageDataTC2, lcVoltage, timePeriod, lcTime)
tc2Sin = np.sin(2*np.pi*timeDataTC2/timePeriod) * voltageDataTC2
_tc2Sin = numpyErrorTcSin(timeDataTC2, lcTime, voltageDataTC2, lcVoltage, timePeriod, lcTime)


sumTc1Cos = np.sum(tc1Cos) - 0.5 * (tc1Cos[0]+tc1Cos[-1])
_sumTc1Cos = np.sqrt(np.sum(np.square(_tc1Cos)) + 0.25*(_tc1Cos[0]**2 + _tc1Cos[-1]**2))
sumTc1Sin = np.sum(tc1Sin) - 0.5 * (tc1Sin[0]+tc1Sin[-1])
_sumTc1Sin = np.sqrt(np.sum(np.square(_tc1Sin)) + 0.25*(_tc1Sin[0]**2 + _tc1Sin[-1]**2))

sumTc2Cos = np.sum(tc2Cos) - 0.5 * (tc2Cos[0]+tc2Cos[-1])
_sumTc2Cos = np.sqrt(np.sum(np.square(_tc2Cos)) + 0.25*(_tc2Cos[0]**2 + _tc2Cos[-1]**2))
sumTc2Sin = np.sum(tc2Sin) - 0.5 * (tc2Sin[0]+tc2Sin[-1])
_sumTc2Sin = np.sqrt(np.sum(np.square(_tc2Sin)) + 0.25*(_tc2Sin[0]**2 + _tc2Sin[-1]**2))


I1Cos = np.sum(tc1Cos)*readingsInterval/timePeriod
_I1Cos = np.sqrt(np.sum(np.square(_tc1Cos*readingsInterval/timePeriod)) + np.square(np.sum(_tc1Cos)*lcTime/timePeriod) + np.square(-1*np.sum(_tc1Cos)*readingsInterval/(timePeriod**2)*lcTime))
I1Sin = np.sum(tc1Sin)*readingsInterval/timePeriod
_I1Sin = np.sqrt(np.sum(np.square(_tc1Sin*readingsInterval/timePeriod)) + np.square(np.sum(_tc1Sin)*lcTime/timePeriod) + np.square(-1*np.sum(_tc1Sin)*readingsInterval/(timePeriod**2)*lcTime))

I2Cos = np.sum(tc2Cos)*readingsInterval/timePeriod
_I2Cos = np.sqrt(np.sum(np.square(_tc2Cos*readingsInterval/timePeriod)) + np.square(np.sum(_tc2Cos)*lcTime/timePeriod) + np.square(-1*np.sum(_tc2Cos)*readingsInterval/(timePeriod**2)*lcTime))
I2Sin = np.sum(tc2Sin)*readingsInterval/timePeriod
_I2Sin = np.sqrt(np.sum(np.square(_tc2Sin*readingsInterval/timePeriod)) + np.square(np.sum(_tc2Sin)*lcTime/timePeriod) + np.square(-1*np.sum(_tc2Sin)*readingsInterval/(timePeriod**2)*lcTime))




Amp1 = np.sqrt(np.square(I1Cos) +np.square(I1Sin)) 
_Amp1 = errorAmp(I1Cos, _I1Cos, I1Sin, _I1Sin)
Amp2 = np.sqrt(np.square(I2Cos) +np.square(I2Sin)) 
_Amp2 = errorAmp(I2Cos, _I2Cos, I2Sin, _I2Sin)


phase1 = np.arctan2(I1Sin, I1Cos) + np.pi
_phase1 = errorPhase(I1Sin, _I1Sin, I1Cos, _I1Cos)
phase2 = np.arctan2(I2Sin, I2Cos) + np.pi
_phase2 = errorPhase(I2Sin, _I2Sin, I2Cos, _I2Cos)



alpha = np.log(Amp2/Amp1)/distanceBetweenRods
_alpha = errorAlpha(Amp1, _Amp1, Amp2, _Amp2, distanceBetweenRods, 0.01)
beta = (phase1 - phase2)/(-1*distanceBetweenRods)
_beta = errorBeta(phase1, _phase1, phase2, _phase2, distanceBetweenRods, 0.01)


D = (2*np.pi/timePeriod)/(2*alpha*beta)
_D = errorD(timePeriod, lcTime, alpha, _alpha, beta, _beta)

print(D, _D, _D/D*100, '%')

df = pd.DataFrame(
    {
        "TC1 Cos": tc1Cos,
        "Error TC1 Cos": _tc1Cos,
        "TC1 Sin": tc1Sin,
        "Error TC1 Sin": _tc1Sin,

        "TC2 Cos": tc2Cos,
        "Error TC2 Cos": _tc2Cos,
        "TC2 Sin": tc2Sin,
        "Error TC2 Sin": _tc2Sin,

        "Sum TC1 Cos": sumTc1Cos,
        "Error Sum TC1 Cos": _sumTc1Cos,

        "Sum TC1 Sin": sumTc1Sin,
        "Error Sum TC1 Sin": _sumTc1Sin,

        "Sum TC2 Cos": sumTc2Cos,
        "Error Sum TC2 Cos": _sumTc2Cos,

        "Sum TC2 Sin": sumTc2Sin,
        "Error Sum TC2 Sin": _sumTc2Sin,

        "I1 Cos": I1Cos,
        "Error I1 Cos": _I1Cos,

        "I1 Sin": I1Sin,
        "Error I1 Sin": _I1Sin,

        "I2 Cos": I2Cos,
        "Error I2 Cos": _I2Cos,

        "I2 Sin": I2Sin,
        "Error I2 Sin": _I2Sin,

        "Amp1": Amp1,
        "Error Amp1": _Amp1,

        "Amp2": Amp2,
        "Error Amp2": _Amp2,

        "Phase1": phase1,
        "Error Phase1": _phase1,

        "Phase2": phase2,
        "Error Phase2": _phase2,

        "Alpha": alpha,
        "Error Alpha": _alpha,

        "Beta": beta,
        "Error Beta": _beta,

        "D": D,
        "Error D": _D


    }
)

df.to_csv("thermal_diffusivity_brass/output.csv", index=False)
print('Sucessfully saved output to output.csv')