import numpy as np
from scipy.io import loadmat 
import os 
from TDsimulation import runSimulation

'''
Script to run time domain simulations using old linear models and simulations available as .mat files.

'''

# get path to current directory
mydir = os.path.dirname(os.path.realpath(__file__))
datapath = mydir + os.sep + 'data' 

# .outb file and pickle file
sim_file = datapath + os.sep + 'simulation_results' + os.sep + 'Step.mat'   # <------Change this
pkl_file = datapath + os.sep +'linear_models' + os.sep + 'ABCD_matrices_old.mat'  # <--------- Change this

# load simulation file
simdata = loadmat(sim_file)
chan = np.array(simdata['chan'])
channame = simdata['channame']

# extract Time,Controls and States
Time = chan[:,0]
controls_ind = [1,5,2]
state_ind = [4,3]

Controls = chan[:,controls_ind]
States = chan[:,state_ind]

# change units of control values to match ones in linear model
Controls = Controls*np.array([1,1000,np.deg2rad(1)])

# change units to match linear models
States = States*np.array([np.deg2rad(1),1/9.5492965964254])

# load linear matrices
ABCD_matrices = loadmat(pkl_file)

# extract
Aw = ABCD_matrices['Aw']
Bw = ABCD_matrices['Bw']
Cw = ABCD_matrices['Cw']
Dw = ABCD_matrices['Dw']

xw = ABCD_matrices['xw']
uw = ABCD_matrices['uw']
yw = ABCD_matrices['yw']

u_h = np.squeeze(ABCD_matrices['u_h'])

debug_ = False

# function to run simulations and plot results
runSimulation(Aw,Bw,Cw,Dw,xw,uw,yw,u_h,Time,States,Controls,debug_)