
#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import pandas as pd
import os, glob
from scipy import stats
from numba import jit, njit, prange


from scipy.signal import savgol_filter

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def GCaMPConvolve(trace, ker):
    if np.sum(trace) == 0:
        return trace
    else:
        trace_conv = np.convolve(trace, ker, 'full')
        trace_conv = trace_conv[1:trace.shape[0]+1] 
        trace_conv[np.logical_not(np.isfinite(trace_conv))] = 0
        trace_conv = trace_conv/max(trace_conv)
        return trace_conv

@njit
def pearsonr_vec_2Dnumb(x,y):
    # computes the pearson correlation coefficient between a a vector (x) and each row in 2d matrix (y), using numba acceleration
    
    n_rows_y = int(y.shape[0])
    corr = np.zeros((n_rows_y))
    for row_y in prange(n_rows_y):
        corr[row_y] = np.corrcoef(x, y[row_y,:])[0,1]
    return corr

#%%
data_dir = os.path.realpath(r'C:\Users\owen.randlett\Dropbox\Written Work\2023_ZebrapmhiPaper\data\20220822')


out_dir = os.path.split(data_dir)[0].replace('data', 'fig_data')
os.chdir(data_dir)
data = np.loadtxt(glob.glob('*coords.txt')[0], delimiter=',')
t_stamps = np.loadtxt(glob.glob('*tstamps.txt')[0], delimiter=',')

stim = t_stamps[1::2]
time = t_stamps[::2]
x_coords = data[::2, :]
x_coords = x_coords - np.nanmean(x_coords[:,0], axis=0)
y_coords = data[1::2, :] 
y_coords = y_coords - np.nanmean(y_coords[:,0], axis=0)


n_frames = min((len(x_coords), len(y_coords)))
x_coords = x_coords[:n_frames]
y_coords = y_coords[:n_frames]

angles = np.arctan2(np.diff(y_coords, axis=1), np.diff(x_coords, axis=1))
angles = np.unwrap(angles)

orients = np.nanmean(angles, axis=1)
diff_angles = np.diff(angles, axis=1)

bend_amps = np.nanmean(diff_angles, axis=1)
bend_amps[np.isnan(bend_amps)] = 0
bend_amps_filt = savgol_filter(bend_amps, 11, 5)


x_ranges = [np.arange(len(stim)), np.arange(312000,313000), np.arange(435000,445000)]
for x_range in x_ranges:
    with plt.rc_context({'lines.linewidth': 2, 'figure.figsize': (12,5), 'font.size':20}):
        plt.plot(x_range, stim[x_range]*0.05)
        plt.plot(x_range, bend_amps_filt[x_range])
        plt.ylim((-0.2,0.2))
        plt.show()

#%%

# load 
ops = np.load(os.path.join(data_dir, 'combined/ops.npy'), allow_pickle=True).item()
F = np.load(os.path.join(data_dir, 'combined/F.npy'), allow_pickle=True)
stat = np.load(os.path.join(data_dir, 'combined/stat.npy'), allow_pickle=True)
iscell = np.load(os.path.join(data_dir, 'combined/iscell.npy'), allow_pickle=True)
stim_df = np.load(os.path.join(data_dir, 'stimvec_mean.npy'))

cell_thresh = 0.3 
cells = iscell[:,1] > cell_thresh
n_cells = np.sum(cells)

F_zscore = stats.zscore(F[cells,:], axis=1)
F_zscore[~np.isfinite(F_zscore)] = 0
stat_cells = stat[cells]

motor_sig = ops['corrXY']
motor_sig[ops['badframes']] = 0
motor_pow = savgol_filter(np.std(rolling_window(motor_sig, 3), -1), 15, 2)
motor_pow = motor_pow - np.median(motor_pow)
motor_pow = motor_pow/np.max(motor_pow)

time_stamps_imaging = np.load(os.path.join(data_dir, 'time_stamps.npy'))
time_stamps_imaging = np.mean(time_stamps_imaging, axis=0)/1000 # time stamp in the middle of the stack, in msec
df_start_inds = np.where(np.diff(stim_df)>0.1)[0]
df_start_inds = np.delete(df_start_inds, np.where(np.diff(df_start_inds) == 1)[0]+1)

df_start = np.where(np.diff(stim.flatten())==-1)[0][0]
t_df_start = time[df_start]

tstamp_imaging_df = time_stamps_imaging[df_start_inds[0]]
tstamp_imaging_end = time_stamps_imaging[-1]

t_imaging_start = t_df_start - tstamp_imaging_df
ind_imagin_start = np.where(time >= t_imaging_start-1.5)[0][0]
t_imaging_end = t_imaging_start + tstamp_imaging_end
ind_imaging_end = np.where(time >= t_imaging_end)[0][0]

bend_amps_filt_imaging = bend_amps_filt[ind_imagin_start:ind_imaging_end]


tail_power = np.std(rolling_window(bend_amps_filt_imaging, int(len(bend_amps_filt_imaging)/len(motor_pow))), -1)
tail_power = tail_power - np.median(tail_power)
bout_thresh = 0.003
bouts = tail_power > 0.003
bout_st_end = np.hstack([np.nan, np.diff(bouts.astype(int))])

    
x_ranges = [np.arange(len(tail_power)), np.arange(505000,510000), np.arange(312000,320000), np.arange(400000,410000)]
for x_range in x_ranges:
    with plt.rc_context({'lines.linewidth': 2, 'figure.figsize': (12,5), 'font.size':20}):
        fig, ax1 = plt.subplots()
        ax1.plot(x_range, tail_power[x_range])
        ax1.plot(x_range, bout_st_end[x_range] * np.mean(tail_power)*7, 'r--')

        ax2 = ax1.twinx()
        ax2.plot(x_range, bend_amps_filt_imaging[x_range], 'C1')
        #plt.ylim((-0.2,0.2))
        plt.show()


bout_st = np.where(bout_st_end == 1)[0]
bout_end = np.where(bout_st_end == -1)[0]
if len(bout_st) > len(bout_end):
    if bout_st[0] > bout_end[0]:
        bout_st = bout_st[1:]
    if  bout_st[-1] >  bout_end[-1]:
        bout_st = bout_st[:-1]

if len(bout_end) > len(bout_st):
    if bout_end[0] < bout_st[0]:
        bout_end = bout_end[1:]
    if bout_end[-1] < bout_st[-1]:
        bout_end = bout_end[:-1]
n_bouts = len(bout_st)
bouts_meanbend = np.zeros(n_bouts)
bouts_power = np.zeros(n_bouts)
for i in range(n_bouts):
    bouts_meanbend[i] = np.nanmean(bend_amps_filt_imaging[bout_st[i]:bout_end[i]]) 
    bouts_power[i] = np.nanmean(tail_power[bout_st[i]:bout_end[i]]) 
#%%
swim_turn_thresh = 0.004
swim_struggle_thresh = 0.017
#%
out_dir = os.path.join(data_dir, 'images_out')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)
with plt.rc_context({'lines.linewidth': 2, 'figure.figsize': (9,12), 'font.size':30}):
    plt.hist(bouts_meanbend, np.arange(-0.02, 0.02, 0.0005))
    plt.vlines([-swim_turn_thresh, swim_turn_thresh], ymin=0, ymax=50, colors=['g', 'm'])
    plt.xlabel('mean tail angle, rad/bout')
    plt.ylabel('counts')
    plt.savefig('tailangle.svg')
    plt.show()

    plt.hist(bouts_power,  np.arange(0.003, 0.04, 0.0005))
    plt.vlines(swim_struggle_thresh,  ymin=0, ymax=200, colors='r')
    plt.xlabel('bout vigor, AU')
    plt.ylabel('counts')
    plt.savefig('tailpower.svg')
    plt.show()

#%%

turn_swim = np.ones(bouts_meanbend.shape) # 1=swim,, 
turn_swim[bouts_meanbend<-swim_turn_thresh] = 2 # 2=turn left ...
turn_swim[bouts_meanbend>swim_turn_thresh] = 3 # 3=turn right (not 100% sure whice on right vs left)

swim_struggle = np.ones(bouts_meanbend.shape) # 1=swim,, 
swim_struggle[bouts_power > swim_struggle_thresh] = 2 # 2=struggle

turn_swim_vec = np.zeros(tail_power.shape)
turn_swim_vec[:] = np.nan

swim_struggle_vec = np.zeros(tail_power.shape)
swim_struggle_vec[:] = np.nan

for i in range(1,4):
    turn_swim_vec[bout_st[turn_swim == i]] = i
    swim_struggle_vec[bout_st[swim_struggle == i]] = i

x_ranges = [np.arange(len(tail_power)), np.arange(505000,510000), np.arange(312000,320000), np.arange(400000,410000)]
for x_range in x_ranges:
    with plt.rc_context({'lines.linewidth': 2, 'figure.figsize': (12,5), 'font.size':20}):
        fig, ax1 = plt.subplots()
        ax1.plot(x_range, tail_power[x_range], label='tail power')
    

        ax2 = ax1.twinx()
        ax2.plot(x_range, bend_amps_filt_imaging[x_range], 'C1')
        ax2.plot(x_range, (turn_swim_vec[x_range]-2) * 0.1, 'r*', label='turn/swim')
        ax2.plot(x_range, (swim_struggle_vec[x_range]-1) * 0.1, 'g*', label='swim/struggle')
        #plt.ylim((-0.2,0.2))
        plt.legend()
        plt.show()

#%%
from scipy import signal, interpolate
tail_power_thresh = 0.1

interp = interpolate.interp1d(time[ind_imagin_start:ind_imaging_end], tail_power)
tail_power_interp = interp(np.arange(time[ind_imagin_start], time[ind_imaging_end], 1/5))
tail_power_frames = signal.resample(tail_power_interp, len(motor_pow))

# interp_bendAmp = interpolate.interp1d(time[ind_imagin_start:ind_imaging_end], bend_amps_filt_imaging)
bend_amps_interp = interp(np.arange(time[ind_imagin_start], time[ind_imaging_end], 1/5))
bend_amps_frames = signal.resample(bend_amps_interp, len(motor_pow))

tail_power_frames[tail_power_frames<tail_power_thresh] = 0
motor_pow[motor_pow<tail_power_thresh] = 0
with plt.rc_context({'font.size':20}):
    max_y = np.max([tail_power_frames, motor_pow])
    fig, ax = plt.subplots(nrows=5, figsize=(13,20))
    
    for k,i in enumerate(range(0, 5)):
        inds = np.arange(1500) + 1500*i
        ax[k].plot(inds, bend_amps_frames[inds]*10)
        ax[k].plot(inds, motor_pow[inds])
        ax[k].set_ylabel('inferred\n movement')
        #ax[k].set_ylim((0,max_y))
        #ax[k].set_ylim((-0.1, 1))
    ax[0].legend(['tail tracking', 'suite2p image\nmotion artifact'])
    
    plt.xlabel('time (frames)')

    plt.show()

# lags = signal.correlation_lags(motor_power_frames.size, motor_pow.size, mode="full")
#%%
time_compression = len(tail_power)/len(motor_pow)
bout_starts_interp = (bout_st/time_compression).astype(int)
bout_ends_interp = (bout_end/time_compression).astype(int)

bout_ends_interp[bout_starts_interp==bout_ends_interp]+=1

swim_turn_vec_frames = np.zeros((3, len(motor_pow)))
for i in range(len(turn_swim)):
    st = bout_starts_interp[i]
    end = bout_ends_interp[i]
    swim_turn_vec_frames[int(turn_swim[i]-1), st:end] = 1

swim_struggle_vec_frames = np.zeros((2, len(motor_pow)))
for i in range(len(swim_struggle)):
    st = bout_starts_interp[i]
    end = bout_ends_interp[i]
    swim_struggle_vec_frames[int(swim_struggle[i]-1), st:end] = 1


# parameters for GCaMP kernel
DecCnst = 0.3
RiseCnst = 0.5
frame_rate = 1.976
DecCnst = DecCnst*frame_rate # now in frames
RiseCnst = RiseCnst*frame_rate

KerRise = np.power(2, (np.arange(0,5)*RiseCnst)) - 1
KerRise= KerRise[KerRise < 1]
KerRise = KerRise/max(KerRise)

KerDec = np.power(2, (np.arange(20, 0, -1)*DecCnst))
KerDec = (KerDec - min(KerDec))/(max(KerDec) - min(KerDec));

KerDec = KerDec[KerDec > 0]
KerDec = KerDec[1:]
KerTotal = np.concatenate([KerRise, KerDec])
# plt.plot(np.arange(len(KerTotal))/frame_rate, KerTotal)
# plt.xlabel('seconds')
# plt.ylabel('predicted GCaMP\nresponse')
# plt.show()

# make regressors with gcamp kernel

regressor_straightswim = GCaMPConvolve(swim_turn_vec_frames[0,:], KerTotal)
regressor_left = GCaMPConvolve(swim_turn_vec_frames[1,:], KerTotal)
regressor_right = GCaMPConvolve(swim_turn_vec_frames[2,:], KerTotal)
regressor_swim = GCaMPConvolve(swim_struggle_vec_frames[0,:], KerTotal)
regressor_struggle = GCaMPConvolve(swim_struggle_vec_frames[1,:], KerTotal)

# inds = np.arange(500,2000)
# plt.plot(regressor_straightswim[inds])
# plt.plot(regressor_left[inds])
# plt.plot(regressor_right[inds])
# plt.plot(regressor_swim[inds])
# plt.plot(regressor_struggle[inds])
#%%
corrMat= np.zeros([n_cells, 5])
corr_order = [
    'straight_swim',
    'left turn',
    'right turn',
    'swim',
    'struggle'
]
corrMat[:, 0] =  pearsonr_vec_2Dnumb(regressor_straightswim, F_zscore)
corrMat[:, 1] =  pearsonr_vec_2Dnumb(regressor_left, F_zscore)
corrMat[:, 2] =  pearsonr_vec_2Dnumb(regressor_right, F_zscore)
corrMat[:, 3] =  pearsonr_vec_2Dnumb(regressor_swim, F_zscore)
corrMat[:, 4] =  pearsonr_vec_2Dnumb(regressor_struggle, F_zscore)
corrMat[np.isnan(corrMat)] = 0
#%%
plt.plot(np.nanmean(F_zscore[corrMat[:, 1] > 0.2, :], axis=0), label='left')
plt.plot(np.nanmean(F_zscore[corrMat[:, 2] > 0.2, :], axis=0), label='right')
plt.legend()
plt.show()

plt.plot(np.nanmean(F_zscore[corrMat[:, 3] > 0.2, :], axis=0), label='swim')
plt.plot(np.nanmean(F_zscore[corrMat[:, 4] > 0.2, :], axis=0), label='struggle')
plt.legend()

#%%


height, width = ops['meanImg'].shape

im_stack = np.zeros((len(corr_order), height, width))
for corr in range(5):
    

    for i in range(n_cells):
        ypix = stat_cells[i]['ypix']
        xpix = stat_cells[i]['xpix']
        # z = stat_cells[i]['iplane']
        im_stack[corr, ypix, xpix] = corrMat[i, corr]
    
    plt.figure(figsize=(20,20))
    plt.imshow(im_stack[corr, :,:], vmin=-0.3, vmax=0.3)
    plt.title(corr_order[corr])
    plt.show()
#%
from tifffile import imsave
imsave('corr_stacks.tif', im_stack)
imsave('anat_image.tif', ops['meanImg'])
#%%

# reconstruct the original stack
n_col = 3
n_row = 4
n_sl = int(n_col*n_row)
height_sl = int(height/n_row)
width_sl = int(width/n_col)


im_slice_stack = np.zeros((n_sl, height_sl, width_sl, len(corr_order)))


corr = 0
for corr in range(5):
    k=0
    for i in range(n_row):
        row_slice = im_stack[corr, height_sl*i:height_sl*(i+1), :]
        for j in range(n_col):
            im = row_slice[:, width_sl*j:width_sl*(j+1)]
            # plt.imshow(im)
            # plt.title(k)
            # plt.show()
            im_slice_stack[k, :,:,corr] = im
            
            k+=1

    plt.imshow(np.mean(im_slice_stack[:,:,:,corr], axis=0))
    plt.title(corr_order[corr])
    plt.show()

    imsave('corr_stacks_slices' + corr_order[corr] + '.tif', im_slice_stack[:,:,:,corr])

#%%

plt.plot(bend_amps_frames)
plt.show()
for key in ops.keys():
    print(key)
# %%
cells = iscell[:,0] == 1
F_cell = F[cells]

plt.imshow(F_cell)