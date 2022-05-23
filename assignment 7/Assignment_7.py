    # EE20B123
    # SHEKHAR DIWAKAR
    # CIRCUIT ANALYSIS USING SYMPY

from sympy import *                                         # Symbolic python module for variables
import pylab as py
import numpy as np
import scipy.signal as sp

s=symbols('s')                                              # Laplace transform variable

# Defifning Low Pass Filter
def lowpass(R1,R2,C1,C2,G,Vi):
    
    A=Matrix([[0,0,1,-1/G],                                 # Conductance Matrix
              [-1/(1+s*R2*C2),1,0,0],
              [0,-G,G,1], 
              [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    
    b=Matrix([0,0,0,Vi/R1])                                 # Source Matrix

    V = A.inv()*b                                           # Variable Matrix
    return (A,b,V)
    
# Defining High Pass Filter    
def highpass(R1,R3,C1,C2,G,Vi):                     
     
     A = Matrix([[0,0,1,-1/G],                              # Conductance Matrix
                 [0,G,-G,-1],
                 [s*C2*R3/(1+s*C2*R3),-1,0,0],
                 [(s*C2+s*C1+1/R1),-s*C2,0,-1/R1]])
                 
     b = Matrix([0,0,0,s*C1*Vi])                            # Source Matrix
     
     V = A.inv()*b
     return (A,b,V)                                         # Source Matrix
     
# Function for converting SymPy symbols to Polynomials     
def Symbols_to_Polynomial(H):
	n,d=fraction(simplify(H))
	return (np.array(Poly(n,s).all_coeffs(),dtype=float),np.array(Poly(d,s).all_coeffs(),dtype=float))
     

# Question_1

A1,b1,V1 = lowpass(10000,10000,1e-9,1e-9,1.586,1)           # Finding the Tansfer Function of low pass filter
Nr_H1,Dr_H1 = Symbols_to_Polynomial(V1[3])
H1 = sp.lti(Nr_H1,Dr_H1)                                    # Converting Sympy Function to Signal Transfer Function
t  = np.linspace(0,0.01,50001)                              # Time steps for plotting the input and the output

Vi1 = np.ones(50001)                                        # Unit Step input function
Vi1[0] = 0
t,Vo1,svec = sp.lsim(H1,Vi1,t)                              # Finding the output in time domain

                   
py.figure(0)                                                 
py.plot(t,abs(Vo1),label = 'Output')                        # plotting the input and the output  
py.plot(t,Vi1,label = 'Input')                            
py.legend(loc = 'upper right')
py.title(r'Q_1:Unit Step Response of Low-Pass Filter')
py.grid(True)

# Question_2 

Vi2 = np.sin(2*(10**3)*(np.pi)*t)+np.cos(2*(10**6)*(np.pi)*t)                            # Sinusoidally oscillating input to the Low pass filter
t,Vo2,svec = sp.lsim(H1,Vi2,t)                                                           # Convolution of the input with the Impulse Response

py.figure(1)                                                                             # plotting the input and the output
py.plot(t,Vi2,label = 'Input')
py.plot(t,-Vo2,label = 'Output')
py.legend(loc = 'upper right')
py.title(r'Q_2:Response for Sinusoidal Input for Low-Pass Filter')
py.grid(True)



# Question_3

A3,b3,V3 = highpass(10000,10000,1e-9,1e-9,1.586,1)                                       # Finding the Tansfer Function of high pass filter 
Vo3 = V3[3]
w  = py.logspace(0,8,801)                                                                # Range of Frequencies for the function to be plotted
ss = 1j*w
hf = lambdify(s,Vo3,'numpy')                                                             # converting the Sympy variable into python function
v  = hf(ss)

py.figure(2)
py.loglog(w,abs(v),lw=2,label = 'Magnitude Response of an High-Pass Filter')             # Plotting the Magnitude Response of High pass filter
py.title(r'Q_3:Magnitude Response of a High-Pass Filter')
py.grid(True)



# Question_4

t4 = np.linspace(0,1,100000)                                                             # Time steps for plotting the input and the output
Vi4 = (np.cos(2*(np.pi)*(10**4)*t4))*(np.exp(-5*t4))                                     # Sinusoudall Damped Oscillating input for high pass filter

Nr_H4,Dr_H4 = Symbols_to_Polynomial(Vo3)                                                 # Converting Sympy Variable to Signal Transfer Function
H4 = sp.lti(Nr_H4,Dr_H4)

t4,Vo4,svec = sp.lsim(H4,Vi4,t4)                                                         # Convolution of the input with the Impulse Response

py.figure(3) 
py.plot(t4,Vi4,label = 'Input freq = 10^4Hz')                                            # plotting the input and the output  
py.plot(t4,Vo4,label = 'Output')
py.legend(loc = 'upper right')
py.title(r'Q_2:Response for Damped-Sinusoidal Input for High-Pass Filter')
py.grid(True)


# Question_5

A5,b5,V5 = highpass(10000,10000,1e-9,1e-9,1.586,1/s)                                     # Finding the Step Response of the high pass filter
Vo5 = V5[3]
t5  = np.linspace(0,0.001,50001)                                                           # Time steps for plotting the input and the output

Vi5 = np.ones(50001)                                                                     # Unit Step input function                                                          
Vi5[0] = 0
t5,Vo5,svec = sp.lsim(H4,Vi5,t5)                                                           # Convolution of Unit step with impulse response of high pass filter

py.figure(4) 
py.plot(t5,abs(Vo5),color='black',label = 'Output')                                       # plotting the input and the output
py.plot(t5,Vi5,label = 'Input')
py.legend(loc = 'upper right')
py.title(r'Q_1:Unit Step Response of High-Pass Filter')
py.grid(True)
py.show()













