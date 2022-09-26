import os
import numpy as np
from pCrunch.io import load_FAST_out
import pickle 
from TDsimulation import runSimulation

'''
Script to run time domain simulations using new linear models and simulations obtained through weis.
Please run this file in your weis-env as you need to use pCrunch to access the simulation data

'''

def extractChannels(outputs,reqd_channels):
    
    '''
    function to extract required channels
    '''
    
    # get the number of channels
    noutputs = len(reqd_channels)
    
    # get the number of time points
    nt = len(outputs['Time'])
    
    # initialize
    Channels = np.zeros((nt,noutputs))
    ind = 0
    
    # loop through and extract channels 
    for chan in reqd_channels:
        Channels[:,ind] = outputs[chan]
        ind+=1
        
    return Channels


# get path to current directory
mydir = os.path.dirname(os.path.realpath(__file__))
datapath = mydir + os.sep + 'data' 

# .outb file and pickle file
sim_file = datapath + os.sep + 'simulation_results' + os.sep + 'step.outb'   
pkl_file = datapath + os.sep +'linear_models' + os.sep + 'ABCD_matrices_new.pkl'  

# load simulations
outputs = load_FAST_out(sim_file)[0]

# required channels
reqd_channels = ['Time','Wind1VelX','GenTq','BldPitch1','PtfmPitch','GenSpeed','TwrBsFxt']
control_ind = [1,2,3]
state_ind = [4,5]

# extract values
Channels = extractChannels(outputs,reqd_channels)

# extract time
Time = Channels[:,0]-100
t0 = Time[0]; tf = Time[-1]

# extract controls
Controls = Channels[:,1:4]

# change units of control values to match ones in linear model
Controls = Controls*np.array([1,1000,np.deg2rad(1)])


# extract states
States = Channels[:,4:6]

# change units to match linear models
States = States*np.array([np.deg2rad(1),1/9.5492965964254])


# load ABCD matrices
with open(pkl_file,'rb') as handle:
    ABCD_matrices = pickle.load(handle)[0]

# extract matrices and transpose them for their shape be (nw,na,nb) 
Aw = ABCD_matrices['A']; Aw = np.transpose(Aw,(2,0,1))
Bw = ABCD_matrices['B']; Bw = np.transpose(Bw,(2,0,1))
Cw = ABCD_matrices['C']; Cw = np.transpose(Cw,(2,0,1))
Dw = ABCD_matrices['D']; Dw = np.transpose(Dw,(2,0,1))

xw = ABCD_matrices['x_ops']
uw = ABCD_matrices['u_ops']
yw = ABCD_matrices['y_ops']

u_h = ABCD_matrices['u_h']

DescOutput = ABCD_matrices['DescOutput']
DescStates = ABCD_matrices['DescStates']
DescCntrlInpt = ABCD_matrices['DescCntrlInpt']

debug_ = True 


runSimulation(Aw,Bw,Cw,Dw,xw,uw,yw,u_h,Time,States,Controls,debug_)

