import numpy as np
import pandas as pd
import uproot
import optimize as optimize
from glob import glob

def getLivetime(fThetaSqrCut,fZetaBDTCut,fFilenames):

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
        print("WARNING! Problems reading the trees in the offruns file " + fname + ", please check that these variables are present: 'RunNr', 'RaSystem', 'DecSystem', 'RaEvent', 'DecEvent', 'CorrEnergy', 'ZetaBDT'! If just a few files are corrupt, the optimization process should work fine.")
    data_bg = data_bg.append(df)
data_bg.set_index('RunNr', inplace=True)
off_run_ids = data_bg.index.unique()
# What is being done here? Just livetime being read?
off_run_data = optimize.read_run_data(off_run_ids)
off_run_data['Livetime'] = off_run_data['Duration'] * (1. - off_run_data['Deadtime_mean'])
# Angle between reconstructed event and pointing direction
data_bg['Psi'] = optimize.angle_between(data_bg['RaEvent'], data_bg['DecEvent'], data_bg['RaSystem'], data_bg['DecSystem'])

#The fuck is this grid for?
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
    print('    -> Run {} / {}'.format(i+1, len(off_run_ids)))

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

    print(ang_size_off)

    # compute livetime-weighted sum
    try:
        bg_size_off += ang_size_off * off_run_data['Livetime'][runid]
    except:
        continue

# basically this whole calculation is giving the average angle used for the bg, we need to get this correct for comparing the same region for gamma MC? Or just for alpha?
bg_size_off /= bg_livetime 
bg_size_off_copy = bg_size_off

print(bg_size_off, bg_livetime)
