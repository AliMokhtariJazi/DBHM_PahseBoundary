#!/usr/bin/env python3

import math
from turtle import color
from xml.dom.expatbuilder import FragmentBuilderNS
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["text.usetex"] = True
plt.rcParams['axes.linewidth'] = 2
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import pandas as pd
##############################################################################################################
def compute_nmi(mu):
    if (mu < 0):
        return 0.0
    elif ( mu % 1 != 0 ):
        return math.ceil(mu/1)
    else:
        return (mu/1) 
##############################################################################################################



##############################################################################################################
def get_E(n, mu):
    return 0.5*n*(n-1)-n*mu
##############################################################################################################



##############################################################################################################
def get_Eplus(mu):
    nmi = compute_nmi(mu)
    return get_E(nmi+1, mu)-get_E(nmi, mu)
##############################################################################################################



##############################################################################################################
def get_Eminus(mu):
    nmi = compute_nmi(mu)
    return get_E(nmi-1, mu)-get_E(nmi, mu)
##############################################################################################################



##############################################################################################################
def compute_z0(mu,Temp):
    num_n = 10
    z0_value = 0
    nmi = compute_nmi(mu)
    for n in range(num_n):
        n_value = np.exp(-(get_E(n, mu)- get_E(nmi, mu))/Temp)
        z0_value = n_value + z0_value
    return z0_value
##############################################################################################################


 
##############################################################################################################
def compute_atomic_g12r(mu,Temp):
    w = 0
    num_n = 10
    z0_value = 0
    atomic_g12r =  0
    nmi = compute_nmi(mu)
    for n in range(num_n):
        x_value = (((n+1)/(w-get_E(n+1, mu)+get_E(n, mu)))-(n/(w+get_E(n-1, mu)-get_E(n, mu))))
        n_value = (np.exp(-(get_E(n, mu)-get_E(nmi, mu))/Temp))* x_value
        z0_value = n_value + z0_value
    atomic_g12r = z0_value / compute_z0(mu,Temp)

    return atomic_g12r
##############################################################################################################



##############################################################################################################
def compute_u1(mu,Temp):
    num_n = 10
    z0_value = 0
    u1_value = 0
    nmi = compute_nmi(mu)
    for n in range(num_n):
        Enmi = get_E(nmi, mu)
        En4 = get_E(n+2, mu)
        En3 = get_E(n+1, mu)
        En2 = get_E(n, mu)
        En1 = get_E(n-1, mu)
        En0 = get_E(n-2, mu)
        x_value1 = (((n+1) * (n+2)/((En4-En2) * ((En3-En2)**2))) + ((n) * (n-1)/((En0-En2) * ((En1-En2)**2))))
        x_value2 = (- ((n+1) * (n+1)/ ((En3-En2)**3)) - (((n) * (n)/ ((En1-En2)**3))))
        x_value3 = (- ((n) * (n+1)/((En3-En2) * ((En1-En2)**2))) - ((n) * (n+1)/((En1-En2) * ((En3-En2)**2))))
        n_value = (np.exp(-(En2-Enmi)/Temp)) * (x_value1 + x_value2 + x_value3)
        z0_value = n_value + z0_value
    u1_value = -2 * (compute_atomic_g12r(mu,Temp)**(-4)) * z0_value / compute_z0(mu,Temp)

    return u1_value
##############################################################################################################




##############################################################################################################
def gen_epsilon_array(dim, L, J):
    kaxis = np.arange(-L/2+1, L/2+1) * (2 * np.pi / L)
    if dim == 1:
        return -2.0 * J * np.cos(kaxis)
    else:
        kgrid = np.meshgrid(*([kaxis]*dim))
        return np.sum(-2.0 * J * np.cos(kgrid), axis=0)
##############################################################################################################




##############################################################################################################
def B_array(mu , sig):
    Em = get_Eminus(mu)
    Ep = get_Eplus(mu)
    return  -(Ep - Em) - sig
############################################################################################################## 



##############################################################################################################
def C_array(mu , sig, Temp):
    ag = compute_atomic_g12r(mu,Temp)
    return  -(1+mu)* (sig - 1/ag)
##############################################################################################################



##############################################################################################################
def exitation_energy(mu , sig, Temp):
    b_k = B_array(mu , sig)
    c_k = C_array(mu , sig, Temp)
    deltaEp = 0.5 * (-b_k+np.sqrt(abs((b_k**2)-(4*c_k))))
    deltaEm = 0.5 * (b_k+np.sqrt(abs((b_k**2)-(4*c_k))))
    return np.array([deltaEp, deltaEm])
##############################################################################################################



##############################################################################################################
def spec_wght(mu, exite_energy):
    denom = exite_energy[0]+exite_energy[1]
    Zplus = ((1+mu)+exite_energy[0])/denom
    Zminus = ((1+mu)-exite_energy[1])/denom
    return np.array([Zplus, Zminus])
 ##############################################################################################################



##############################################################################################################
def n_k_array(spectral_weight):
    return 0.5*(spectral_weight[0]+spectral_weight[1]-1)
##############################################################################################################



################################################################################################
def sig_12_R(dim , L , J, mu, n_now, Temp, delta):
    nmi = compute_nmi(mu)
    epsilon = gen_epsilon_array(dim, L, J)
    u1 = compute_u1(mu, Temp)
    g_12 = compute_atomic_g12r(mu,Temp)
    delta_bar = delta/8/(np.log2(2))
    G12 = g_12 + (g_12**2)*(epsilon + 2*u1*(n_now-nmi))
    return (epsilon
            + 2 * u1 * (n_now-nmi)
            + (3/2)*(delta_bar**2)*(G12)
            )
################################################################################################



################################################################################################
def calc_n_now_HBF(J, mu, dim , L , n_guess, Temp, delta,tolerance ,alpha):
    n_now = n_guess
    while True:
        Nsite = (L)**(dim)
        sig = sig_12_R(dim , L , J, mu, n_now, Temp, delta)
        exite_energy = exitation_energy(mu , sig, Temp)
        spectral_weight = spec_wght(mu, exite_energy)
        n_k = n_k_array(spectral_weight)
        n_old = np.sum(n_k)/Nsite

        g_12 = compute_atomic_g12r(mu,Temp)
        G12_k = -1/(sig -(1/g_12))
        G12 = np.sum(G12_k)/Nsite

        if (abs(n_old - n_now) > tolerance):
            n_now = (n_now*alpha  + n_old*(1-alpha) )
            # print(n_now, n_old)
        else:
            n_now = n_old
            break
    return [n_now, G12]
################################################################################################



##############################################################################################################
def phi_function(mu , J, dim , L , n_guess, Temp, delta,tolerance ,alpha):
    nmi = compute_nmi(mu)
    g_12 = compute_atomic_g12r(mu,Temp)
    u1 = compute_u1(mu,Temp)
    delta_bar = delta/8/(np.log2(2))
    data = calc_n_now_HBF(J, mu, dim , L , n_guess, Temp, delta,tolerance ,alpha)
    n_now = data[0]
    G_12 = data[1]
    G12 = np.average(G_12)
    if (delta==0.0):
        return (2*u1*(n_now-nmi) - (1/g_12))/2/dim
    else:
        return (-2*dim*J + 2*u1*(n_now-nmi)- (1/g_12)
                + (6)*(delta_bar**2)*G12
                + (9)*(delta_bar**4)*G12*(g_12**2)
                # + 2*(18+3/2)*(delta_bar**2)*(g_12**2)*u1*(n_now-nmi)
                )
##############################################################################################################

################################################################################################
def calc_J_start(mu,delta,dim):
    if (dim==1):
        Ji = 0


    if (dim==2):
        if (delta==0.1):
            if(mu<0.42):
                y0 = 0.07
                Ji = (mu - y0)*0.045/(0.42-y0)
            else:
                y0 = 0.9
                Ji = (mu - y0)*0.045/(0.42-y0) 


        if (0.16<delta<=0.23):
            if(mu<0.42):
                y0 = 0.11
                Ji = (mu - y0)*0.037/(0.42-y0)
            else:
                y0 = 0.82
                Ji = (mu - y0)*0.037/(0.42-y0)   


        if (delta==0.3):
            if(mu<0.42):
                y0 = 0.18
                Ji = (mu - y0)*0.028/(0.42-y0)
            else:
                y0 = 0.75
                Ji = (mu - y0)*0.028/(0.42-y0)    


        if (delta==0.4):
            if(mu<0.42):
                y0 = 0.22
                Ji = (mu - y0)*0.017/(0.42-y0)
            else:
                y0 = 0.62
                Ji = (mu - y0)*0.017/(0.42-y0)  


        if (delta==0.5):
            Ji = 0
        
        else:
            Ji = 0



    if (dim==3):
        if (delta==0.1):
            if(mu<0.42):
                y0 = 0.07
                Ji = (mu - y0)*0.029/(0.42-y0)
            else:
                y0 = 0.9
                Ji = (mu - y0)*0.029/(0.42-y0) 


        if (delta==0.2):
            if(mu<0.42):
                y0 = 0.11
                Ji = (mu - y0)*0.025/(0.42-y0)
            else:
                y0 = 0.82
                Ji = (mu - y0)*0.025/(0.42-y0)   


        if (delta==0.3):
            if(mu<0.42):
                y0 = 0.18
                Ji = (mu - y0)*0.018/(0.42-y0)
            else:
                y0 = 0.75
                Ji = (mu - y0)*0.018/(0.42-y0)    


        if (delta==0.4):
            if(mu<0.42):
                y0 = 0.22
                Ji = (mu - y0)*0.010/(0.42-y0)
            else:
                y0 = 0.62
                Ji = (mu - y0)*0.010/(0.42-y0)  


        if (delta==0.5):
            Ji = 0
        else:
            Ji = 0



    if (Ji>=0):
        return Ji
    else:
        return 0
################################################################################################

# print(calc_J_start(0.4,0.1,2))


################################################################################################
def J_fun(mu , dim , L , n_guess, Temp, delta,tolerance ,alpha):
    # J= 0.0
    if (dim==1):
        if (0.02<delta<=0.11):
            step = 0.0001 
            if (0.03<mu<0.94):
                tolerance = 0.001
                Jf = 0.16
            else:
                tolerance = 0.01
                Jf = 0.003


        if (0.11<delta<=0.16):
            Jf = 0.15
            tolerance = 0.001
            step = 0.0003


        if (0.16<delta<=0.21):
            Jf = 0.11
            tolerance = 0.001
            step = 0.0003
            if (0.07<mu<0.86):
                tolerance = 0.001
                Jf = 0.14
            else:
                tolerance = 0.0001
                Jf = 0.005


        if (0.21<delta<=0.26):
            Jf = 0.1
            tolerance = 0.001
            step = 0.0003

        if (0.26<delta<=0.31):
            step = 0.0003
            if (0.12<mu<0.8):
                tolerance = 0.001
                Jf = 0.1
            else:
                tolerance = 0.01
                Jf = 0.005

        if (0.31<delta<=0.36):
            Jf = 0.08
            tolerance = 0.001
            step = 0.0003

        if (0.36<delta<=0.41):
            Jf = 0.05
            tolerance = 0.001
            step = 0.0003

        if (0.41<delta<=0.8):
            Jf = 0.05
            tolerance = 0.001
            step = 0.0003


    if (dim==2):

        if (0.03<delta<=0.11):
            step = 0.00003
            if (0.04<mu<0.94):
                tolerance = 0.001
                Jf = 0.3
            else:
                0.045
                tolerance = 0.001
                Jf = 0.003


        if (0.11<delta<=0.16):
            step = 0.00003
            if (0.05<mu<0.9):
                tolerance = 0.001
                Jf = 0.5
            else:
                0.045
                tolerance = 0.001
                Jf = 0.02


        if (0.16<delta<=0.23):
            step = 0.00003
            if (0.07<mu<0.85):
                tolerance = 4*0.0001
                Jf = 0.045
            else:
                0.045
                tolerance = 0.001
                Jf = 0.005


        if (0.23<delta<=0.26):
            Jf = 0.042
            tolerance = 0.0001
            step = 0.00001

        if (0.26<delta<=0.31):
            Jf = 0.035
            tolerance = 0.0001
            step = 0.00003

        if (0.31<delta<=0.36):
            Jf = 0.03
            tolerance = 0.0001  
            step = 0.00003

        if (0.36<delta<=0.41):
            Jf = 0.025
            tolerance = 0.0001 
            step = 0.00003

        if (0.41<delta<=0.8):
            Jf = 0.02
            tolerance = 0.0001
            step = 0.00003


    if (dim==3):
        if (0.03<delta<=0.11):
            step = 0.0003
            tolerance = 0.001
            Jf = 0.035


        if (0.11<delta<=0.16):
            Jf = 0.035
            tolerance = 0.001
            step = 0.0003

        if (0.16<delta<=0.21):
            Jf = 0.03
            tolerance = 4*0.001
            step = 0.0001


        if (0.21<delta<=0.26):
            Jf = 0.025
            tolerance = 0.001
            step = 0.0001

        if (0.26<delta<=0.31):
            Jf = 0.025
            tolerance = 3*0.0001
            step = 0.0001

        if (0.31<delta<=0.36):
            Jf = 0.020
            tolerance = 3*0.0001
            step = 0.0001

        if (0.36<delta<=0.41):
            Jf = 0.015
            tolerance = 3*0.0001
            step = 0.0001

        if (0.41<delta<=0.9):
            Jf = 0.015
            tolerance = 3*0.0001
            step = 0.0001

    critical2 = 1e+15
    J = calc_J_start(mu,delta,dim)
    while True:
        Critical = phi_function(mu , J, dim , L , n_guess, Temp, delta,tolerance ,alpha)
        if(abs(Critical)>1*tolerance) and (Jf + 0.0003 > J):
            J = J+step
            # print(J , "    ", Critical)
            if (J>=Jf):
                Jc = -0.02
                break
            
        else:
            Jc = J
            break
        # if  (abs(Critical)<abs(critical2))and (Jf + 0.0003 > J):
        #     J = J+step
        #     critical2 = Critical
        #     # print(J , "    ", Critical)
        # else:
        #     Jc = J
        #     break
    print(Jc)    
    return Jc
##############################################################################################################
# J_fun(0.419 , 2 , 100 , 1, 1e-10, 0.209,1e-6 ,0.9)

################################################################################################
def calc_phase_boundary_HFB(L, dim, step_size, n_guess, Temp, delta, nlobes,tolerance ,alpha):
    J_array = []
    mu_array = []
    if (0.03<delta<=0.11):
        yy = 0.02
        xx = 0.06
    if (0.11<delta<=0.24):
        yy = 0.05
        xx = 0.1
    if (0.24<delta<=0.31):
        if(dim==2):
            yy = 0.12
        else:
            yy = 0.1
        xx = 0.2
    if (0.31<delta<=0.41):
        yy = 0.17
        xx = 0.28
    if (0.41<delta<=0.51):
        yy = 0.23
        xx = 0.4
    if (0.51<delta):
        yy = 0.23
        xx = 0.4
    else:
        print("Input is not valid")


    for lobe in range(nlobes):
        mu = lobe + yy+(step_size/2)
        while (mu < lobe+1.0-(step_size/2)-xx):
            J = J_fun(mu , dim , L , n_guess, Temp, delta,tolerance ,alpha)
            # if (0.41 < mu < 0.42):
            #     Jc = J
            print("mu is:          ",mu, "        J is:      ",J, "         Delta is:         ",delta)
            J_array.append(J)
            mu_array.append(mu)
            mu = mu + step_size

    J_array = np.array(J_array)
    mu_array = np.array(mu_array)
    # n_now_array = np.array(n_now_array)

    return [J_array, mu_array]
################################################################################################



# J_fun(0.3 , 1 , 100 , 1, 1e-12, 0.1, 1e-6 ,0.99)
##############################################################################################################
##############################################################################################################
##############################################################################################################
def Compute_J(n_now, mu, dim, Temp, delta,G12):
    nmi = compute_nmi(mu)
    g_12 = compute_atomic_g12r(mu,Temp)
    u1 = compute_u1(mu,Temp)
    delta_bar = delta/8/(np.log2(2))

    if (delta==0.0):
        return (2*u1*(n_now-nmi) - (1/g_12))/2/dim
    else:
        return (2*u1*(n_now-nmi)- (1/g_12)
                + (6)*(delta_bar**2)*G12
                + (9)*(delta_bar**4)*G12*(g_12**2)
                # + 2*(18+3/2)*(delta_bar**2)*(g_12**2)*u1*(n_now-nmi)
                )/2/dim




def Critical_HFB1(mu, dim , L , n_guess, Temp, delta):
    Jc = 0.0
    g_12 = compute_atomic_g12r(mu,Temp)
    G12 = g_12
    n_now = n_guess
    n_old = n_now
    while True:
        Nsite = (L)**(dim)
        # el = int(L/2)-1
        J = Compute_J(n_now, mu, dim, Temp, delta , G12)
        sig = sig_12_R(dim , L , J, mu, n_now, Temp, delta)
        G12_k = -1/(sig -(1/g_12))
        G12 = np.sum(G12_k)/Nsite
        exite_energy = exitation_energy(mu , sig, Temp)
        spectral_weight = spec_wght(mu, exite_energy)

        n_k = n_k_array(spectral_weight)
        n_now = np.sum(n_k)/Nsite

        if (delta==0.0):
            tolerance = 1e-6
            alpha = 0.5
        else:
            tolerance = 1 * 1e-4
            alpha = 0.01

        if ((n_now - n_old)**2 > tolerance):
            n_old = (n_now * alpha + n_old * (1-alpha))
            # print(n_now,"  ",n_old)
        else:
            if (0.41 < mu < 0.42):

                Jc = J
            break
    return [n_now , J, Jc]




def calc_phase_boundary_HFB1(L, dim, step_size, n_guess, Temp, delta, nlobes):
    J_array = []
    mu_array = []
    J_c_array = []
    n_now_array = []
    for lobe in range(nlobes):
        mu = lobe + 0.999999
        while (mu >lobe + 0.0):
            CriticalHFB1 = Critical_HFB1(mu, dim , L , n_guess+lobe, Temp, delta)
            n_now = CriticalHFB1[0]
            J = CriticalHFB1[1]
            Jc = CriticalHFB1[2]
            # print("mu is:          ",mu, "        J is:      ",J,"            n_now is:      ", n_now)
            J_array.append(J)
            J_c_array.append(Jc)
            mu_array.append(mu)
            n_now_array.append(n_now)
            n_guess= n_now
            # n_guess = (n_now * alpha + n_guess * (1-alpha))
            mu = mu - step_size

    Jcc = np.sum(J_c_array)
    # print(Jcc)
    J_array = np.array(J_array)
    mu_array = np.array(mu_array)
    n_now_array = np.array(n_now_array)
    return [J_array, mu_array, n_now_array,Jcc]
##############################################################################################################
##############################################################################################################
##############################################################################################################



##############################################################################################################
def plot_phase_boundary(dim, L, n_guess,Temp,tolerance ,alpha):
    Delta_Jc_array = []
    U_J_array = []
    delta_array =  [0.0, 0.05,0.1,0.2,0.3,0.4,0.5]
    for delta in delta_array:
        if (delta==0.0):
            res= calc_phase_boundary_HFB1(L, dim, 0.01, n_guess, Temp, delta, 1)
            Jc = res[3]
            print(delta, Jc)
        else:
            Jc= J_fun(0.4199 , dim , L , n_guess, Temp, delta,tolerance ,alpha)
            print(delta, delta/Jc, 1/Jc)
        Delta_Jc_array.append(delta/Jc)
        U_J_array.append(1/Jc)
    print(Jc,1/Jc)
    plt.plot(U_J_array, Delta_Jc_array,  '--', linewidth=1, markersize=3 ,label="Effective Theory")  
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(6,0.5,'v', linewidth=0.5)

    if (dim==1):
        plt.xlim((0.0 , 6.6))
        plt.ylim((0.0,4))
        # plt.text(0.286,0.05, '(a)', fontsize=20, ha='center',
        # bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
    if (dim==2):
        data = pd.read_csv("Default_Dataset.csv")
        Xx_axis = data['X_axis'].tolist()
        Yy_axis = data['Y_axis'].tolist()
        plt.plot(Xx_axis,Yy_axis, '-o',color ='red', markersize=6,label="QMC Simulation")

        data2 = pd.read_csv("Default_Dataset_Super.csv")
        Xxs_axis = data2['X_axis'].tolist()
        Yys_axis = data2['Y_axis'].tolist()
        plt.plot(Xxs_axis,Yys_axis, '-o',color ='red', markersize=2)

        data3 = pd.read_csv("Default_Dataset_Super1.csv")
        Xxs1_axis = data3['X_axis'].tolist()
        Yys1_axis = data3['Y_axis'].tolist()
        plt.plot(Xxs1_axis,Yys1_axis, '--',color ='red', markersize=2)

        plt.text(46,8, 'MI', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.text(20,35, 'SF', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.text(53,45, 'BG', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.xlim((0.0 , 60))
        plt.ylim((0.0,80))


    if (dim==3):
        data = pd.read_csv("Default_Dataset_3d.csv")
        Xx_axis = data['X_axis'].tolist()
        Yy_axis = data['Y_axis'].tolist()
        plt.plot(Xx_axis,Yy_axis, '-o',color ='red', markersize=2,label="QMC Simulation")


        data2 = pd.read_csv("Default_Dataset_Super_3D.csv")
        Xxs_axis = data2['X_axis'].tolist()
        Yys_axis = data2['Y_axis'].tolist()
        plt.plot(Xxs_axis,Yys_axis, '-o',color ='red', markersize=2)

        plt.text(170,30, 'MI', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.text(30,125, 'SF', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.text(170,230, 'BG', fontsize=20, ha='center',
        bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
        plt.xlim((0.0 ,200 ))
        plt.ylim((0.0,400))


    plt.xlabel(r"$U/J$", fontsize=20)
    plt.ylabel(r"$ \Delta/J $", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(loc="upper right", prop={"size":16})
    plt.tight_layout()
    plt.savefig(f'ppttrlot_phase_boundary_{dim}_mu=0.42.pdf')
    plt.show()
    return 0
##############################################################################################################

# plot_phase_boundary(2, 100, 1,1e-12,1e-5 ,0.9)



################################################################################################
def plot_phase_boundary(dim, L, n_guess, step_size,Temp, nlobes,tolerance ,alpha):
    Maximum_pic = 0
    # ,0.2,0.3,0.4,0.5
    delta_array = [0.4]
    for delta in delta_array:
        if (delta==0.0):
            calc_phase = calc_phase_boundary_HFB1(L, dim, step_size, n_guess, Temp, delta, nlobes)
        else:
            calc_phase = calc_phase_boundary_HFB(L, dim, step_size, n_guess, Temp, delta, nlobes,tolerance ,alpha)
        plt.plot(calc_phase[0], calc_phase[1],  'v', markersize=2, label=f'$\Delta/U$ = {delta}')  
        # print(calc_phase[2])
        Maximum_pic = max(Maximum_pic, max(calc_phase[0]))
    if (dim==1):
        plt.xlim(0, 0.3)
        plt.text(0.284,0.06, '(a)', fontsize=20, ha='center',
            bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
    if (dim==2):
        plt.xlim(0, 0.08)
        plt.text(0.076,0.06, '(b)', fontsize=20, ha='center',
            bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
    if (dim==3):
        plt.xlim(0.0002, 0.055)
        plt.text(0.0524,0.06, '(c)', fontsize=20, ha='center',
            bbox=dict( fc="w", ec="w", alpha=1, lw=0.25))
    plt.ylim(0,nlobes)
    plt.xlabel(r"$J/U$", fontsize=20)
    plt.ylabel(r"$ \mu/U $", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(loc="upper right", prop={"size":15},frameon=False)
    plt.tight_layout()
    plt.savefig(f'yyyplot_phase_boundary2_{dim}D_delta4.pdf')
    plt.show()
    return 0
################################################################################################


plot_phase_boundary(2, 100, 1, 0.01,1e-8, 1,1e-5 ,0.9)


################################################################################################
def Experiment_comparison(dim, L, n_guess, step_size,Temp, nlobes,tolerance ,alpha):
    Maximum_pic = 0
    # ,0.2,0.3,0.4,0.5
    # 2.17
    # 0.0409
    # ,0.217
    delta_array = [0.209,0.225]
    for delta in delta_array:
        if (delta==0.0):
            calc_phase = calc_phase_boundary_HFB1(L, dim, step_size, n_guess, Temp, delta, nlobes)
        else:
            calc_phase = calc_phase_boundary_HFB(L, dim, step_size, n_guess, Temp, delta, nlobes,tolerance ,alpha)
        plt.plot(calc_phase[0], calc_phase[1],  '--', markersize=2, label=f'$\Delta/U$ = {delta}')  
        plt.plot(0.0409,0.42,  'o', markersize=5)  
        # print(calc_phase[2])
        Maximum_pic = max(Maximum_pic, max(calc_phase[0]))
    if (dim==1):
        plt.xlim(0, 0.3)
    if (dim==2):
        plt.xlim(0, 0.08)
    if (dim==3):
        plt.xlim(0.0002, 0.055)
    plt.ylim(0,nlobes)
    plt.xlabel(r"$J/U$", fontsize=20)
    plt.ylabel(r"$ \mu/U $", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(loc="upper right", prop={"size":15},frameon=False)
    plt.tight_layout()
    plt.savefig(f'plot_phase_boundary2_{dim}D.pdf')
    plt.show()
    return 0
################################################################################################

# Experiment_comparison(2, 100, 1, 0.01,1e-8, 1 ,1e-5 ,0.9)