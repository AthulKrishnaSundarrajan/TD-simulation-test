import os
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from scipy.interpolate import PchipInterpolator,CubicSpline
from scipy.integrate import solve_ivp

   
def odefun(t,x,A_op,B_op,W_fun,U_fun,Uo_fun,DXoDt_fun,Wavg,caseflag):
    
    '''
     Derivative functions
    '''
    
    if caseflag == 1:
        '''
        LTI simulation, using a single linear model at wind speed w = w_{avg} of the simulation:
            
        dx = A*x + B*(u(t) - u_op(w(t)))
        '''        
        # average wind value
        w = Wavg
        
        # AB matrices
        A = A_op(w)
        B = B_op(w)
        u = U_fun(t)
        Uo = Uo_fun(w)
        
        xd = np.dot(A,x) + np.dot(B,(u-Uo))
        
    elif caseflag == 2:
        '''
        LPV simulation, using all the linear models
            
        dx = A(w)*x + B(w)*(u(t) - u_op(w(t))) - dx_op(w)/dt
        '''  
        
        #lpv model
        w = W_fun(t)
        
        # AB matrices
        A = A_op(w)
        B = B_op(w)
        u = U_fun(t)
        Uo = Uo_fun(w)
        
        dXoDt = DXoDt_fun(t)
        
        xd = np.dot(A,x) + np.dot(B,(u-Uo)) - dXoDt.T
   
    return xd
        
        
        

def runSimulation(Aw,Bw,Cw,Dw,xw,uw,yw,u_h,Time,States,Controls,debug_):  
    
    ''' 
    Function to construct the LPV model and run constraints
    '''
    
    # t0 and tf
    t0 = Time[0]; tf = Time[-1]
    
    # wind speed
    Wind = Controls[:,0]
    Wavg  = np.mean(Wind)
    
    # create interpolating function to get U(t)
    U_pp = PchipInterpolator(Time,Controls,axis = 0)
    U_fun = lambda t: U_pp(t)
    
    Controls_ = Controls.copy()
    Controls_[:,2] = Controls_[:,2] + np.random.rand(len(Time))/100

    Uo_debug_pp = PchipInterpolator(Time,Controls_)
    Uo_debug = lambda t: Uo_debug_pp(t)
    
    # interpolating function for wind speed
    W_pp = CubicSpline(Time,Wind)
    
    # get derivative of wind function dW/dt
    dW_pp = W_pp.derivative 
    dW_pp = dW_pp(nu = 1)
    
    # create anonymous function
    DW_fun = lambda t: dW_pp(t)
    W_fun = lambda t: W_pp(t)

    # A matrix interpolating function
    A_op_pp = PchipInterpolator(u_h,Aw,axis = 0)
    A_op = lambda w: A_op_pp(w)
    
    # B matrix interpolating function
    B_op_pp = PchipInterpolator(u_h,Bw,axis = 0)
    B_op = lambda w: B_op_pp(w)
    
    # C matrix interpolating function
    C_op_pp = PchipInterpolator(u_h,Cw,axis = 0)
    C_op = lambda w: C_op_pp(w)
    
    # D matrix interpolating function
    D_op_pp = PchipInterpolator(u_h,Dw,axis = 0)
    D_op = lambda w: D_op_pp(w)
    
    # control operating points interpolating function
    Uo_pp = PchipInterpolator(u_h,uw,axis = 1,extrapolate = True)
    Uo_fun = lambda w: Uo_pp(w)
       
    
    # state operating points interpolating function
    Xo_pp = CubicSpline(u_h, xw, axis = 1,extrapolate = True)
    Xo_fun = lambda w: Xo_pp(w)
    
    # outputs interpolating function
    Yo_pp = PchipInterpolator(u_h, yw, axis = 1,extrapolate = True)
    Yo_fun = lambda w: Yo_pp(w)
    
    # first time derivative of state operating points
    DXo_pp = Xo_pp.derivative 
    DXo_pp = DXo_pp(nu=1)
    DXo_fun = lambda w: DXo_pp(w)
    
    # offset term
    DXoDt_fun = lambda t: (-DXo_fun(W_fun(t)).T*DW_fun(t)).T
    
    '''
    compare U_fun(t) against Uo_fun(t)) 
    '''


    if debug_:
        
        # evaluate quantiites
        U_ = U_fun(Time)
        Uop_ = Uo_fun(W_fun(Time)).T



        # initialize plot 
        fig, axa = plt.subplots(1)
        fig, axb = plt.subplots(1)
        fig, axc = plt.subplots(1)
    
        # wind
        axa.plot(Time,U_[:,0],'k',label ='OpenFAST')
        axa.plot(Time,Uop_[:,0],'r',label = 'Uop')
        axa.set_title('Wind Speed [m/s]')
        axa.set_xlim([t0,tf])
        axa.legend()
    
        # torue
        axb.plot(Time,U_[:,1],'k',label ='OpenFAST')
        axb.plot(Time,Uop_[:,1],'r',label = 'Uop')
        axb.set_title('Gen Torque [MWm]')
        axb.set_xlim([t0,tf])
        axb.legend()
    
        # blade pitch
        axc.plot(Time,U_[:,2],'k',label ='OpenFAST')
        axc.plot(Time,Uop_[:,2],'r',label = 'Uop')
        axc.set_title('Bld Pitch [rad]')
        axc.set_xlim([t0,tf])
        axc.legend()
    

    # get shape of matrices
    nw,nx,nu = np.shape(Bw)

    # initial states
    X0 = np.zeros((nx))
    
    X0[0] = States[0,0]
    X0[4] = States[0,1] 
    
    # time span for simulation
    tspan = [t0,tf]

    '''
    LTI simulation
    '''
    caseflag = 1
    
    # intial states
    X0avg = X0 - Xo_fun(Wavg)
    
    # simulate
    sol = solve_ivp(odefun,tspan,X0avg,method = 'RK45',args = (A_op,B_op,W_fun,U_fun,Uo_fun,DXoDt_fun,Wavg,caseflag))
    
    # extract results
    Tavg = sol['t']
    Xavg = sol['y']
    
    # add offset
    Xavg = Xavg.T + Xo_fun(Wavg)
    
    '''
    LPV simulation
    '''
    caseflag = 2
    
    # correct initial states
    X0lpv = X0 - Xo_fun(W_fun(0))
    
    # solve
    sol = solve_ivp(odefun,tspan,X0lpv,method = 'RK45',args = (A_op,B_op,W_fun,U_fun,Uo_fun,DXoDt_fun,Wavg,caseflag))
    
    # extract
    Tlpv = sol['t']
    Xlpv = sol['y']
    
    # add offset
    Xlpv = (Xlpv + Xo_fun(W_fun(Tlpv)))
    Xlpv = Xlpv.T
    
    '''
    Plot results
    '''
    
    # platform pitch
    fig1,ax1 = plt.subplots(1,1)
    ax1.set_ylabel('PtfmPitch [deg]',fontsize = 16)
    ax1.set_xlabel('Time [s]',fontsize = 16)
    ax1.set_xlim([t0,tf])
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    ax1.plot(Time,np.rad2deg(States[:,0]),label = 'OpenFAST')
    ax1.plot(Tavg,np.rad2deg(Xavg[:,0]),label = 'LTI')
    ax1.plot(Tlpv,np.rad2deg(Xlpv[:,0]),label = 'LPV')
    ax1.legend()
    
    # generator speed
    fig2,ax2 = plt.subplots(1,1)
    ax2.set_ylabel('GenSpeed [rad/s]',fontsize = 16)
    ax2.set_xlabel('Time [s]',fontsize = 16)
    ax2.set_xlim([t0,tf])
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    ax2.plot(Time,States[:,1],label = 'OpenFAST')
    ax2.plot(Tavg,Xavg[:,4],label = 'LTI')
    ax2.plot(Tlpv,Xlpv[:,4],label = 'LPV')
    ax2.legend()

    plt.show()




