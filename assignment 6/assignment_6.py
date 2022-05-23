#APL ASSIGNMENT 6
#EE20B123
#SHEKHAR DIWAKAR

# importing modules.
from pylab import *
import scipy.signal as sp

# The time response for decay 0.5

F1 = poly1d([1, 0.5]) 
F_1 = polymul([1, 1, 2.5], [1, 0, 2.25])
X1 = sp.lti(F1, F_1)  
t1, x1 = sp.impulse(X1, None, linspace(0, 50, 1000))

figure(1)
plot(t1, x1)
title(" X(t) for decay of 0.5",fontsize=12)
xlabel(r'$t\rightarrow$',fontsize=10)
ylabel(r'$x(t)\rightarrow$',fontsize=10)
grid(True)

# The time response for higher decay i.e 0.05

F2 = poly1d([1, 0.05])
F_2 = polymul([1, 0.1, 2.2525],[1, 0, 2.25])
X2 = sp.lti(F2, F_2)
t2, x2 = sp.impulse(X2, None, linspace(0,50,1000))


figure(2)
plot(t2, x2)
title("X(t) for smaller decay of 0.05",fontsize=12)
xlabel(r'$t\rightarrow$',fontsize=10)
ylabel(r'$x(t)\rightarrow$',fontsize=10)
grid(True)


H = sp.lti([1],[1, 0, 2.25])  
for omega in arange(1.4, 1.6, 0.05):  # omega vary from 1.4 to 1.6 with a step of 0.05
	t3 = linspace(0, 50, 1000)
	f = cos(omega*t3)*exp(-0.05*t3)
	t, x, svec = sp.lsim(H, f, t3)


	figure(3)
	plot(t, x, label = 'omega = ' + str(omega))
	title("x(t) for different frequencies( omega range from 1.4 to 1.6)",fontsize=12)
	xlabel(r'$t\rightarrow$',fontsize=10)
	ylabel(r'$X(t)\rightarrow$',fontsize=10)
	legend(loc = 'upper left')
	grid(True)
	

# solving coupled spring equation and finding time response function

t4 = linspace(0, 20, 1000)
H4_X = sp.lti(np.poly1d([1, 0, 2]), poly1d([1, 0, 3, 0]))
X4 = sp.impulse(H4_X, T=t4)
H4_Y = sp.lti(np.poly1d([2]), poly1d([1, 0, 3, 0]))
Y4 = sp.impulse(H4_Y, T=t4)

# plotes for time reponses
figure(4)
plot(X4[0], X4[1], label = 'x(t)')
plot(Y4[0], Y4[1], label = 'y(t)')
title("X(t) and Y(t)",fontsize=12)
xlabel(r'$t\rightarrow$',fontsize=10)
ylabel(r'$functions X(t) and Y(t)\rightarrow$',fontsize=10)
legend(loc = 'upper right')
grid(True)

#To find the Transfer Equation of two port network  and plotting bode plots of magnitude and phase.
w = 1.5
zeta = 0
R = 100
L = 1e-6
C = 1e-6

w = 1/sqrt(L*C) 
Q = 1/R * sqrt(L/C)
zeta = 1/(2*Q)

F3 = poly1d([w**2])
F_3 = poly1d([1, 2*w*zeta, w**2])

H = sp.lti(F3, F_3)

w, S, phi = H.bode()

figure(5)
semilogx(w, S)
title("Magnitude Bode plot",fontsize=12)
xlabel(r'$\omega\rightarrow$',fontsize=10)
ylabel(r'$20\log|H(j\omega)|\rightarrow$',fontsize=10)
grid(True)

figure(6)
semilogx(w, phi)
title("Phase Bode plot",fontsize=12)
xlabel(r'$\omega\rightarrow$',fontsize=10)
ylabel(r'$\angle H(j\omega)\rightarrow$',fontsize=10)
grid(True)

t6 = arange(0, 30e-6, 1e-8)
Vi = cos(1e3*t6) - cos(1e6*t6)
t6, Vo, svec = sp.lsim(H, Vi, t6)

#To Find the output voltage from transfer function and input voltage for short term and long term time intervals.The plots of output voltages.
	
figure(7)
plot(t6, Vo)
title("The Output Voltage for short time interval",fontsize=12)
xlabel(r'$t\rightarrow$',fontsize=10)
ylabel(r'$V_o(t)\rightarrow$',fontsize=10)
grid(True)

t7 = arange(0, 10e-3, 1e-8)
Vi = cos(1e3*t7) - cos(1e6*t7)
t7, V_o, svec = sp.lsim(H, Vi, t7)

figure(8)
plot(t7, V_o)
title("The Output Voltage for long time interval",fontsize=12)
xlabel(r'$t\rightarrow$',fontsize=10)
ylabel(r'$V_o(t)\rightarrow$',fontsize=10)
grid(True)
show()