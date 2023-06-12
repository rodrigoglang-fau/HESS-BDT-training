import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import figure
from scipy import interpolate # for effective area interpolation

import sys,os
import scripts.optimize as optimize
from glob import glob

import uproot

def getLivetime(fThetaSqrCut,fZetaBDTCut,fFilenames): # adapted from Lars' optimization script

    # Reading exclusion regions
    ers = optimize.ExclusionRegionSet()
    ers.read_from_file('/home/extern/rglang/modules/hessuser/hess_hap/head/hdanalysis/lists/ExcludedRegions.dat')
    ers.read_from_file('/home/extern/rglang/modules/hessuser/hess_hap/head/hdanalysis/lists/ExcludedRegions-stars.dat')

    # List with off files
    f_off = sorted(glob('classes-hybrid/hap/Optimization-Offruns_std_zeta_hybrid_2d3_20deg_0deg/events*.root'))
    data_bg = pd.DataFrame()
    # Just reading the trees from the off files
    for fname in f_off:
        ev_file = uproot.open(fname)
        #ev_tree = ev_file['ParTree_PMBgMaker_Off']
        ev_tree = ev_file['ParTree_Preselect']
        try:
            df = ev_tree.arrays(['RunNr', 'RaSystem', 'DecSystem', 'RaEvent', 'DecEvent', 'CorrEnergy', 'ZetaBDT'], library='pd')
        except:
            print("WARNING! Problems reading the trees in the offruns file " + fname + ", please check that these variables are present: 'RunNr', 'RaSystem', 'DecSystem', 'RaEvent', 'DecEvent', 'CorrEnergy','ZetaBDT'! If just a few files are corrupt, the optimization process should work fine.")
        data_bg = data_bg.append(df)
    data_bg.set_index('RunNr', inplace=True)
    off_run_ids = data_bg.index.unique()
    # What is being done here? Just livetime being read?
    off_run_data = optimize.read_run_data(off_run_ids)
    off_run_data['Livetime'] = off_run_data['Duration'] * (1. - off_run_data['Deadtime_mean'])
    # Angle between reconstructed event and pointing direction
    data_bg['Psi'] = optimize.angle_between(data_bg['RaEvent'], data_bg['DecEvent'], data_bg['RaSystem'], data_bg['DecSystem'])

    grid_size = 1.2
    grid_vals = 1201
    offset_val = np.linspace(-0.5*grid_size, 0.5*grid_size, grid_vals)
    offset_x, offset_y = np.meshgrid(offset_val, offset_val, indexing='ij')

    # Masks with event within give offset direction.
    bg_psi_masks = []
    bg_rundata = []
    bg_livetime = off_run_data['Livetime'].sum()
    bg_size_off = 0

    for i,runid in enumerate(off_run_ids):
        #print('    -> Run {} / {}'.format(i+1, len(off_run_ids)))

        # rundata apparently contains only the pointing direction for each run
        rundata = data_bg.loc[runid]
        ra_pnt = rundata['RaSystem'].values[0]
        dec_pnt = rundata['DecSystem'].values[0]
        bg_rundata.append(rundata)

        # get exclusion regions within 1 degree of pointing position
        run_ers = ers.get_regions_within(ra_pnt, dec_pnt, 1.0)

        # select events in annulus, store mask
        m_bg = (rundata['Psi'] > 0.5-np.sqrt(fThetaSqrCut)) & (rundata['Psi'] < 0.5+np.sqrt(fThetaSqrCut))
        m_bg &= ~run_ers.contains(rundata['RaEvent'].values, rundata['DecEvent'].values)[0]
        bg_psi_masks.append(m_bg)

        # be lazy and compute size of ring minus exclusion regions numerically -> calculating the effective area on the annulus taking out the deadtime events
        grid_ra = ra_pnt + offset_x / np.cos(dec_pnt * np.pi / 180)
        grid_dec = dec_pnt + offset_y
        grid_offset = optimize.angle_between(grid_ra, grid_dec, ra_pnt, dec_pnt)
        m_grid = (grid_offset > 0.5-np.sqrt(fThetaSqrCut)) & (grid_offset < 0.5+np.sqrt(fThetaSqrCut))
        m_grid &= ~run_ers.contains(grid_ra, grid_dec)[0]
        ang_size_off = grid_size**2 * m_grid.sum() / grid_vals**2

        # compute livetime-weighted sum
        try:
            bg_size_off += ang_size_off * off_run_data['Livetime'][runid]
        except:
            continue

    # basically this whole calculation is giving the average angle used for the bg which is needed for alpha
    bg_size_off /= bg_livetime 
    bg_size_off_copy = bg_size_off

    return(bg_size_off, bg_livetime)

def saveRawData(X,Y,outputName,descX,descY):

    fout = open(outputName,'w')

    fout.write("#" + descX + "," + descY + '\n')

    for i in range(len(X)):
        fout.write(str(X[i]) + "," + str(Y[i]))
        if i < len(X)-1:
            fout.write('\n')

    fout.close()

    return

def getLiAndMa(fNOn, fNOff, fAlpha): # returns the Li&Ma significance for a given number of ON and OFF counts and alpha
    
    fFirstTerm = fNOn * np.log((1+fAlpha)*fNOn / (fAlpha*(fNOn+fNOff)))
    fSecondTerm = fNOff * np.log((1+fAlpha)*fNOff / (fNOn + fNOff) )

    if (fFirstTerm < 0):
        return(-1)
    
    return(np.sqrt(2 * (fFirstTerm+fSecondTerm)))

def getPsi(RAEvent,DecEvent,RASystem,DecSystem): # calculates the angles between the reconstructed event position and the center of the telescope (pointing direction)
    Psi = []
    for i in range(len(RAEvent)):
        phi1 = RAEvent[i] * np.pi/180.
        phi2 = RASystem[i] * np.pi/180.
        theta1 = DecEvent[i] * np.pi/180.
        theta2 = DecSystem[i] * np.pi/180.
        ax1 = np.cos(phi1)  * np.cos(theta1)
        ay1 = np.sin(-phi1) * np.cos(theta1)
        az1 = np.sin(theta1)
        ax2 = np.cos(phi2)  * np.cos(theta2)
        ay2 = np.sin(-phi2) * np.cos(theta2)
        az2 = np.sin(theta2)
        res = np.arccos(np.clip(ax1*ax2 + ay1*ay2 + az1*az2, -1, 1))       
        Psi.append(res * 180 / np.pi)
    Psi=np.array(Psi)
    return(Psi) # in deg

def getIRFPlots(workDirectory, config, muonPhase, zenith, azimuth, offset, thetaSqrCut, zetaBDTCut, outputName):

    #plotting style

    mpl.rcParams['figure.figsize'] = (12,10)
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['axes.labelsize'] = 22 
    mpl.rcParams['axes.titlesize'] = 22 
    mpl.rcParams['lines.linewidth'] = 3        
    mpl.rcParams['hist.bins'] = 40
    mpl.rcParams['legend.fontsize'] = 18



    print("Reading files")
    offsetString = str(offset).replace('.', 'd')
    signalFilename = workDirectory + "/hap/Optimization-Gamma_" + config + "_" + muonPhase + "_" + str(zenith) + "deg_" + str(azimuth) + "deg_" + str(offsetString) + "deg_events.root"
    backgroundFilename = workDirectory + "/hap/Optimization-Offruns_" + config + "_" + muonPhase + "_" + str(zenith) + "deg_" + str(azimuth) + "deg_events.root"

    if (not os.path.exists(signalFilename)):
        sys.exit("ERROR! Non-existent input file: " + signalFilename)

    if (not os.path.exists(backgroundFilename)):
        sys.exit("ERROR! Non-existent input file: " + backgroundFilename)

    signalFile = uproot.open(signalFilename)
    backgroundFile = uproot.open(backgroundFilename)

    try:
        signalTree = signalFile["ParTree_Preselect"]
    except:
        print("ERROR! No ParTree_Preselect tree found on gamma file, please check if the file is correct!")

    try:
        backgroundTree = backgroundFile["ParTree_PMBgMaker_Off"]
    except:
        try:
            backgroundTree = backgroundFile["ParTree_Preselect"]
        except:
            print("ERROR! No ParTree_Preselect or ParTree_PMBgMaker_Off tree found on background file, please check if the file is correct!")

    print("Calculating effective area")

    binsEnergy = signalFile["ThrownEnergy"].axis().edges() # Use the same as ThrownEnergy or that /3 or what?
    simulatedEvents = signalFile["ThrownEnergy"].values()
    simulatedArea = signalFile["AreaThrown"].values()[0] / 11 # Needs to be compensated by 1/nentries...

    energy = signalTree["Energy"].array()
    MCTrueEnergy = signalTree["MCTrueEnergy"].array()
    zetaBDT = signalTree["ZetaBDT"].array()
    thetaSqr = signalTree["MCThetaSqr"].array()

    effectiveAreaPreselect = {}
    effectiveAreaZetaBDT = {}
    effectiveAreaZetaBDTAndThetaSqr = {}

    for e in ["Energy","MCTrueEnergy"]:
        effectiveAreaPreselect[e] = np.zeros(len(binsEnergy)-1)
        effectiveAreaZetaBDT[e] = np.zeros(len(binsEnergy)-1)
        effectiveAreaZetaBDTAndThetaSqr[e] = np.zeros(len(binsEnergy)-1)

    effectiveAreaX = []
    for i in range(len(binsEnergy)-1):
        effectiveAreaX.append((binsEnergy[i+1]+binsEnergy[i])/2)
    
    for i in range(len(energy)):
        if (energy[i] > 0):
            binEnergy = int((np.log10(energy[i])-np.log10(binsEnergy[0])) / (np.log10(binsEnergy[1])-np.log10(binsEnergy[0])))
            binMCEnergy = int((np.log10(MCTrueEnergy[i])-np.log10(binsEnergy[0])) / (np.log10(binsEnergy[1])-np.log10(binsEnergy[0])))
            if (binEnergy >= 0 and binEnergy < len(effectiveAreaPreselect["Energy"])-1):
                effectiveAreaPreselect["Energy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]
                if (zetaBDT[i] < zetaBDTCut):
                    effectiveAreaZetaBDT["Energy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]
                    if (thetaSqr[i] < thetaSqrCut):
                        effectiveAreaZetaBDTAndThetaSqr["Energy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]
            if (binMCEnergy >= 0 and binMCEnergy < len(effectiveAreaPreselect["Energy"])-1):
                effectiveAreaPreselect["MCTrueEnergy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]
                if (zetaBDT[i] < zetaBDTCut):
                    effectiveAreaZetaBDT["MCTrueEnergy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]
                    if (thetaSqr[i] < thetaSqrCut):
                        effectiveAreaZetaBDTAndThetaSqr["MCTrueEnergy"][binEnergy] += simulatedArea*1./simulatedEvents[binMCEnergy]

    interpolatedEffectiveArea = {}
    interpolatedEffectiveArea["Energy"] = interpolate.interp1d(effectiveAreaX, effectiveAreaZetaBDTAndThetaSqr["Energy"])
    interpolatedEffectiveArea["MCTrueEnergy"] = interpolate.interp1d(effectiveAreaX, effectiveAreaZetaBDTAndThetaSqr["MCTrueEnergy"])

    saveRawData(effectiveAreaX,effectiveAreaPreselect["Energy"],outputName+"-EffectiveArea-RecEnergy-Preselect.dat","Reconstructed energy [TeV]","Effective area [m^2]")
    saveRawData(effectiveAreaX,effectiveAreaZetaBDT["Energy"],outputName+"-EffectiveArea-RecEnergy-ZetaBDT.dat","Reconstructed energy [TeV]","Effective area [m^2]")
    saveRawData(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["Energy"],outputName+"-EffectiveArea-RecEnergy-ZetaBDTAndThetaSqr.dat","Reconstructed energy [TeV]","Effective area [m^2]")
    saveRawData(effectiveAreaX,effectiveAreaPreselect["MCTrueEnergy"],outputName+"-EffectiveArea-TrueEnergy-Preselect.dat","True energy [TeV]","Effective area [m^2]")
    saveRawData(effectiveAreaX,effectiveAreaZetaBDT["MCTrueEnergy"],outputName+"-EffectiveArea-TrueEnergy-ZetaBDT.dat","True energy [TeV]","Effective area [m^2]")
    saveRawData(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["MCTrueEnergy"],outputName+"-EffectiveArea-TrueEnergy-ZetaBDTAndThetaSqr.dat","True energy [TeV]","Effective area [m^2]")

    plt.loglog(effectiveAreaX,effectiveAreaPreselect["Energy"],label="Preselect")
    plt.loglog(effectiveAreaX,effectiveAreaZetaBDT["Energy"],label="ZetaBDT")
    plt.loglog(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["Energy"],label="ZetaBDT + ThetaSqr")
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel(r"$A_{\mathrm{eff}} \ \left[\mathrm{m^2}\right]$")
    plt.ylim([3e3,1e6])
    plt.legend(loc="best")
    plt.title(config)
    plt.savefig(outputName + '-EffectiveArea-1.png')
    plt.savefig(outputName + '-EffectiveArea-1.pdf')
    plt.clf()

    plt.loglog(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["Energy"])
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel(r"$A_{\mathrm{eff}} \ \left[\mathrm{m^2}\right]$")
    plt.ylim([3e3,1e6])
    plt.title(config)
    plt.savefig(outputName + '-EffectiveArea-2.png')
    plt.savefig(outputName + '-EffectiveArea-2.pdf')
    plt.clf()

    plt.loglog(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["MCTrueEnergy"])
    plt.xlabel(r"$E_{\mathrm{true}}$ [TeV]")
    plt.ylabel(r"$A_{\mathrm{eff}} \ \left[\mathrm{m^2}\right]$")
    plt.ylim([3e3,1e6])
    plt.title(config)
    plt.savefig(outputName + '-EffectiveArea-3.png')
    plt.savefig(outputName + '-EffectiveArea-3.pdf')
    plt.clf()

    plt.loglog(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["Energy"],label=r"$E_{\mathrm{rec}}$")
    plt.loglog(effectiveAreaX,effectiveAreaZetaBDTAndThetaSqr["MCTrueEnergy"],label=r"$E_{\mathrm{true}}$")
    plt.xlabel(r"$E$ [TeV]")
    plt.ylabel(r"$A_{\mathrm{eff}} \ \left[\mathrm{m^2}\right]$")
    plt.ylim([3e3,1e6])
    plt.legend(loc="best")
    plt.title(config)
    plt.savefig(outputName + '-EffectiveArea-4.png')
    plt.savefig(outputName + '-EffectiveArea-4.pdf')
    plt.clf()

    print("Calculating energy dispersion, resolution and bias")

    binsEnergy = np.logspace(-2,2,40)

    zetaBDT = np.array(signalTree["ZetaBDT"].array())
    MCThetaSqr = np.array(signalTree["MCThetaSqr"].array())
    energy = np.array(signalTree["Energy"].array())[zetaBDT < zetaBDTCut] # Only zetaBDT or also theta sqr??
    MCTrueEnergy = np.array(signalTree["MCTrueEnergy"].array())[zetaBDT < zetaBDTCut]

    energyDispersion = np.histogram2d(MCTrueEnergy,(energy-MCTrueEnergy)/MCTrueEnergy,bins=(binsEnergy,np.linspace(-3,3,300)))
    energyBiasX = []
    energyBiasY = []
    energyResolutionY = []

    for i in range(len(energyDispersion[0])):
        if (energyDispersion[0][i].sum() > 0):
            energyBiasX.append((energyDispersion[1][i+1]+energyDispersion[1][i])/2)
            mean = 0
            for j in range(len(energyDispersion[0][i])):
                mean += energyDispersion[0][i][j] * energyDispersion[2][j] / energyDispersion[0][i].sum()
            energyBiasY.append(mean*100) # in %
            stdDevSquared = 0
            for j in range(len(energyDispersion[0][i])):
                stdDevSquared += energyDispersion[0][i][j] * np.power(energyDispersion[2][j] - mean,2) / energyDispersion[0][i].sum()
        
            energyResolutionY.append(np.sqrt(stdDevSquared)*100)

    plt.hist2d(MCTrueEnergy,(energy-MCTrueEnergy)/MCTrueEnergy,bins=(binsEnergy,np.linspace(-0.75,0.75,75)),cmin = 1)
    plt.xscale('log')
    plt.xlabel(r"$E_{\mathrm{true}}$ [TeV]")
    plt.ylabel(r"$\left(E_{\mathrm{rec}} - E_{\mathrm{true}}\right)/E_{\mathrm{true}}$")
    plt.title(config)
    plt.savefig(outputName + '-EnergyDispersion.png')
    plt.savefig(outputName + '-EnergyDispersion.pdf')
    plt.clf()

    plt.plot(energyBiasX,energyBiasY)
    plt.show()
    plt.xscale('log')
    plt.xlabel(r"$E_{\mathrm{true}}$ [TeV]")
    plt.ylabel("Energy bias [%]")
    plt.title(config)
    plt.savefig(outputName + '-EnergyBias.png')
    plt.savefig(outputName + '-EnergyBias.pdf')
    plt.clf()

    saveRawData(energyBiasX,energyBiasY,outputName+"-EnergyBias.dat","True energy [TeV]","Energy bias (%)")
    saveRawData(energyBiasX,energyResolutionY,outputName+"-EnergyResolution.dat","True energy [TeV]","Energy resolution (%)")

    plt.plot(energyBiasX,energyResolutionY)
    plt.xscale('log')
    plt.xlabel(r"$E_{\mathrm{true}}$ [TeV]")
    plt.ylabel("Energy resolution [%]")
    plt.title(config)
    plt.savefig(outputName + '-EnergyResolution.png')
    plt.savefig(outputName + '-EnergyResolution.pdf')
    plt.clf()

    print("Calculating angular resolution")

    binsEnergy = np.logspace(-2,2,25)

    zetaBDT = np.array(signalTree["ZetaBDT"].array())
    MCThetaSqr = np.array(signalTree["MCThetaSqr"].array())[zetaBDT < zetaBDTCut]
    energy = np.array(signalTree["Energy"].array())[zetaBDT < zetaBDTCut]
    angularResolutionX = []
    angularResolutionY = []

    for i in range(len(binsEnergy)-1):
        MCThetaSqrBinned = MCThetaSqr[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])]

        if (len(MCThetaSqrBinned)>5):
        
            MCThetaSqrBinned = np.sort(MCThetaSqrBinned)
        
            angularResolutionX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            angularResolutionY.append(np.sqrt(MCThetaSqrBinned[int(len(MCThetaSqrBinned)*0.68+1)]))

    saveRawData(angularResolutionX,angularResolutionY,outputName+"-AngularResolution.dat","Reconstructed energy [TeV]","Angular resolution, 68% [deg]")

    plt.plot(angularResolutionX,angularResolutionY)
    plt.xscale('log')
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel("Angular resolution, 68% [deg]")
    plt.ylim([0.04,0.1])
    plt.axhline(y=np.sqrt(thetaSqrCut),color="black",linestyle='--',linewidth=1.5)
    plt.text(1e2,np.sqrt(thetaSqrCut)+4e-4,r"$\theta^{2}$ cut",ha='right',size=16)
    plt.title(config)
    plt.savefig(outputName + '-AngularResolution.png')
    plt.savefig(outputName + '-AngularResolution.pdf')
    plt.clf()

    print("Calculating cut efficiency")

    binsEnergy = np.logspace(-2,2,40)

    gammaEfficiencyX=[]
    gammaEfficiencyY=[]
    backgroundEfficiencyX=[]
    backgroundEfficiencyY=[]
    gammaEfficiencyWithThetaSqrX=[]
    gammaEfficiencyWithThetaSqrY=[]
    backgroundEfficiencyWithThetaSqrX=[]
    backgroundEfficiencyWithThetaSqrY=[]

    zetaBDT = np.array(signalTree["ZetaBDT"].array())
    MCThetaSqr = np.array(signalTree["MCThetaSqr"].array())
    energy = np.array(signalTree["Energy"].array())
    energyZetaBDT = np.array(signalTree["Energy"].array())[zetaBDT < zetaBDTCut]
    energyZetaBDTAndThetaSqr = np.array(signalTree["Energy"].array())[np.logical_and(zetaBDT < zetaBDTCut,MCThetaSqr < thetaSqrCut)]

    for i in range(len(binsEnergy)-1):
        if (len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])] > 0)):
            gammaEfficiencyX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            gammaEfficiencyWithThetaSqrX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            gammaEfficiencyY.append( len(energyZetaBDT[np.logical_and(energyZetaBDT < binsEnergy[i+1],energyZetaBDT > binsEnergy[i])]) / len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])]))
            gammaEfficiencyWithThetaSqrY.append( len(energyZetaBDTAndThetaSqr[np.logical_and(energyZetaBDTAndThetaSqr < binsEnergy[i+1],energyZetaBDTAndThetaSqr > binsEnergy[i])]) / len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])]))
        
    zetaBDT = np.array(backgroundTree["ZetaBDT"].array())
    #MCThetaSqr = np.array(backgroundTree["MCThetaSqr"].array()) # There just isn't a proper way to estimate theta sqr from offruns since we don't know the true value
    energy = np.array(backgroundTree["Energy"].array())
    energyZetaBDT = np.array(backgroundTree["Energy"].array())[zetaBDT < zetaBDTCut]
    #energyZetaBDTAndThetaSqr = np.array(backgroundTree["Energy"].array())[np.logical_and(zetaBDT < zetaBDTCut,MCThetaSqr < thetaSqrCut)]

    for i in range(len(binsEnergy)-1):
        if (len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])] > 0)):
            backgroundEfficiencyX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            backgroundEfficiencyWithThetaSqrX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            backgroundEfficiencyY.append( 1 - len(energyZetaBDT[np.logical_and(energyZetaBDT < binsEnergy[i+1],energyZetaBDT > binsEnergy[i])]) / len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])]))
            #backgroundEfficiencyWithThetaSqrY.append( 1 - len(energyZetaBDTAndThetaSqr[np.logical_and(energyZetaBDTAndThetaSqr < binsEnergy[i+1],energyZetaBDTAndThetaSqr > binsEnergy[i])]) / len(energy[np.logical_and(energy < binsEnergy[i+1],energy > binsEnergy[i])]))        

    saveRawData(gammaEfficiencyX,gammaEfficiencyY,outputName+"-SignalEfficiency-ZetaBDT.dat","Reconstructed energy [TeV]","Signal efficiency - zetaBDT")
    saveRawData(gammaEfficiencyWithThetaSqrX,gammaEfficiencyWithThetaSqrY,outputName+"-SignalEfficiency-ZetaBDTAndThetaSqr.dat","Reconstructed energy [TeV]","Signal efficiency - zetaBDT + thetasqr")
    saveRawData(backgroundEfficiencyX,backgroundEfficiencyY,outputName+"-BackgroundEfficiency-ZetaBDT.dat","Reconstructed energy [TeV]","Background cut efficiency - zetaBDT")

    plt.plot(gammaEfficiencyX,gammaEfficiencyY,color="b",linestyle="--",label="ZetaBDT")
    plt.plot(gammaEfficiencyX,gammaEfficiencyWithThetaSqrY,color="b",label=r"ZetaBDT + $\theta^2$")
    plt.xscale('log')
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel("Signal efficiency")
    plt.ylim([0.04,1])
    plt.axhline(y=zetaBDTCut,color="black",linestyle='--',linewidth=1.5)
    plt.text(1e2,zetaBDTCut+1e-2,r"ZetaBDT cut",ha='right',size=16)
    plt.legend(loc="best")
    plt.title(config)
    plt.savefig(outputName + '-SignalEfficiency.png')
    plt.savefig(outputName + '-SignalEfficiency.pdf')
    plt.clf()

    plt.plot(backgroundEfficiencyX,backgroundEfficiencyY,color="r",label="ZetaBDT")
    #plt.plot(backgroundEfficiencyX,backgroundEfficiencyWithThetaSqrY,color="b",linestyle="--")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel("Backgorund cut efficiency")
    plt.legend(loc="best")
    plt.title(config)
    plt.savefig(outputName + '-BackgroundCutEfficiency.png')
    plt.savefig(outputName + '-BackgroundCutEfficiency.pdf')
    plt.clf()

    print("Calculating sensitivity")

    print(" -> Calculating background livetime and angular size")

    backgroundOffSize, backgroundLivetime = getLivetime(thetaSqrCut,zetaBDTCut,workDirectory + '/hap/Optimization-Offruns_' + config + "_" + muonPhase + "_" + str(zenith) + "deg_" + str(azimuth) + "deg/events*root")
    onSize = np.pi * thetaSqrCut
    alpha = onSize/backgroundOffSize
    print("  -> Livetime = {} s | Offsize = {} deg^2 | Alpha = {}".format(backgroundLivetime,backgroundOffSize,alpha))

    print(" -> Reading off data")

    gammaZetaBDT = np.array(signalTree["ZetaBDT"].array())
    gammaMCThetaSqr = np.array(signalTree["MCThetaSqr"].array())
    gammaEnergy = np.array(signalTree["Energy"].array())[np.logical_and(gammaZetaBDT < zetaBDTCut,gammaMCThetaSqr < thetaSqrCut)]

    backgroundZetaBDT = np.array(backgroundTree["ZetaBDT"].array())
    backgroundRAEvent = np.array(backgroundTree["RaEvent"].array())
    backgroundDecEvent = np.array(backgroundTree["DecEvent"].array())
    backgroundRASystem = np.array(backgroundTree["RaSystem"].array())
    backgroundDecSystem = np.array(backgroundTree["DecSystem"].array())
    backgroundPsi = getPsi(backgroundRAEvent,backgroundDecEvent,backgroundRASystem,backgroundDecSystem)
    backgroundEnergy = np.array(backgroundTree["CorrEnergy"].array())[np.logical_and(backgroundZetaBDT < zetaBDTCut,backgroundPsi > 0.5-np.sqrt(thetaSqrCut),backgroundPsi < 0.5+np.sqrt(thetaSqrCut))]

    # Sensitivity criteria -> important!

    minExcess = 10
    minRatio = 0.05
    minSigma = 5
    observationTime = 50 * 3600 #s

    binsEnergy = np.logspace(-2,2,20) # 5 bins per decade

    differentialSensitivityX = []
    differentialSensitivityY = []

    effectiveAreaMaximum=0
    for i in np.logspace(-2,2,400):
        if (interpolatedEffectiveArea["Energy"](i) > effectiveAreaMaximum):
            effectiveAreaMaximum = interpolatedEffectiveArea["Energy"](i)

    print(" -> Obtaining flux for criteria")

    for i in range(len(binsEnergy)-1):
        if (interpolatedEffectiveArea["Energy"]((binsEnergy[i+1]+binsEnergy[i])/2) < effectiveAreaMaximum*0.1): # energy threshold
            continue
        print("E = " + str((binsEnergy[i+1]+binsEnergy[i])/2))
        nOff = len( backgroundEnergy[np.logical_and(backgroundEnergy < binsEnergy[i+1],backgroundEnergy > binsEnergy[i])] ) * observationTime / backgroundLivetime
        if (nOff > 0):
            nOnFor5Sigma = np.maximum(int(nOff*alpha+minExcess),int((1+minRatio)*alpha*nOff))
            significance = -1
            while (significance < minSigma):
                nOnFor5Sigma += 1
                significance = getLiAndMa(nOnFor5Sigma,nOff,alpha)
            binFactor = np.power(1/binsEnergy[i] - 1/binsEnergy[i+1] , -1) # because the simulation is done with E^-2
            spectrumFactor = np.power((binsEnergy[i+1]+binsEnergy[i])/2,-2)
            fluxFor5Sigma = (nOnFor5Sigma-alpha*nOff)*binFactor*spectrumFactor/(observationTime * interpolatedEffectiveArea["Energy"]((binsEnergy[i+1]+binsEnergy[i])/2))
            differentialSensitivityX.append((binsEnergy[i+1]+binsEnergy[i])/2)
            differentialSensitivityY.append(fluxFor5Sigma)

    differentialSensitivityX = np.array(differentialSensitivityX)
    differentialSensitivityY = np.array(differentialSensitivityY)

    saveRawData(differentialSensitivityX,differentialSensitivityY,outputName+"-Sensitivity.dat","Reconstructed energy [TeV]","E^2 x Differential sensitivity [TeV m^-2 s^-1]")

    plt.loglog(differentialSensitivityX,differentialSensitivityX*differentialSensitivityX*differentialSensitivityY)
    plt.xlabel(r"$E_{\mathrm{rec}}$ [TeV]")
    plt.ylabel(r"$E^{2} \times $ Differential sensitivity $\left[ \mathrm{TeV \, m^{-2} \, s^{-1}}  \right]$")
    plt.ylim([3e-10,3e-7])
    plt.title(config)
    plt.savefig(outputName + '-Sensitivity.png')
    plt.savefig(outputName + '-Sensitivity.pdf')
    plt.clf()

