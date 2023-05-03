#%%

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import os, glob
import FishTrack
from scipy.signal import savgol_filter
data_dir = os.path.realpath(r'C:\Users\owen.randlett\Dropbox\Written Work\2023_ZebrapmhiPaper\data\20221213')
out_dir = os.path.split(data_dir)[0].replace('data', 'fig_data')
os.chdir(data_dir)
data = np.loadtxt(glob.glob('*coords.txt')[0], delimiter=',')
t_stamps = np.loadtxt(glob.glob('*tstamps.txt')[0], delimiter=',')
#%%

os.chdir(out_dir)

stim = t_stamps[1::2]
time = t_stamps[::2]
x_coords = data[::2, :]
x_coords = x_coords - np.nanmean(x_coords[:,0], axis=0)
y_coords = data[1::2, :] 
y_coords = y_coords - np.nanmean(y_coords[:,0], axis=0)

n_frames = len(x_coords)
# for tp in range(506, n_frames, int(n_frames/100)):
#     plt.plot(x_coords[tp,:], y_coords[tp,:])
# plt.show()
angles = np.arctan2(np.diff(y_coords, axis=1), np.diff(x_coords, axis=1))
angles = np.unwrap(angles)

orients = np.nanmean(angles, axis=1)
diff_angles = np.diff(angles, axis=1)

# while np.max(np.abs(diff_angles)) > np.pi:
#     angles[np.where(diff_angles > np.pi)[0] + 1] = angles[np.where(diff_angles > np.pi)[0] + 1] - 2*np.pi
#     angles[np.where(diff_angles <-np.pi)[0] + 1] = angles[np.where(diff_angles <- np.pi)[0] + 1] + 2*np.pi
#     #
# diff_angles = np.diff(angles, axis=1)

bend_amps = np.nanmean(diff_angles, axis=1)
bend_amps[np.isnan(bend_amps)] = 0
bend_amps_filt = savgol_filter(bend_amps, 11, 5)

cmap = cm.get_cmap('CMRmap_r')

bracket = 350
n_frames_plot = bracket
y_lim_bend = 45
n_xtick =4

elevation = 40

def make_track_plots(frame):
    st = frame
    end = frame + bracket
    #x = np.arange(st, end)
    x = time[st:end]

    for i in range(len(x)-1):
        plt.plot(x[i:i+2],np.rad2deg(orients[st+i:st+i+2]), color = cmap(i/n_frames_plot) )

    plt.xlabel('time elapsed (sec)')
    plt.ylabel('average tail angle (deg)')
    plt.ylim([-y_lim_bend,y_lim_bend])
    plt.xlim([x[0], x[-1]])
    plt.gca().set_facecolor((0.8,0.8,0.8))
    # plt.vlines(x[int(len(x)/2)], -30, 30, 'r', linestyles='dashed')
    # plt.vlines(x[n_frames_plot], -30, 30, 'r', linestyles='dashed')

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(n_xtick))
    plt.savefig("tail_angle_frame_" + str(st) + '.svg')
    plt.show()

    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(projection='3d')
    for k, tp in reversed(list(enumerate(range(frame, frame+n_frames_plot)))):
        ax.plot(x_coords[tp,:-1], y_coords[tp,:-1], time[tp], color = cmap(k/n_frames_plot), alpha=0.75)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elevation, azim=-45)
    ax.set_ylim([-20,20])
    ax.set_xlim([0,70])
    ax.invert_zaxis()
    # make the panes transparent
    ax.xaxis.set_pane_color((0.8, 0.8, 0.8, 1))
    ax.yaxis.set_pane_color((0.8, 0.8, 0.8, 1))
    ax.zaxis.set_pane_color((0.8, 0.8, 0.8, 1))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_zlabel('time elapsed (sec)')
    plt.savefig("tail_track_3d_" + str(st) + '.svg')
    plt.show()

with plt.rc_context({'lines.linewidth': 2, 'figure.figsize': (12,5), 'font.size':20}):
    # struggle event at 690 seconds,
    make_track_plots(71420)
    make_track_plots(33140)
    make_track_plots(54415)

#%%



#%%
diff_first_angle = angles[:,1:].copy()
#%%
for i in range(diff_first_angle.shape[1]-1):
    diff_first_angle[:,i] = diff_first_angle[:, i] - angles[:,0]

while np.max(np.abs(diff_first_angle)) > np.pi:
    angles[np.where(diff_first_angle > np.pi)[0] + 1] = angles[np.where(diff_first_angle > np.pi)[0] + 1] - 2*np.pi
    angles[np.where(diff_first_angle <-np.pi)[0] + 1] = angles[np.where(diff_first_angle <- np.pi)[0] + 1] + 2*np.pi
    diff_first_angle = np.diff(angles)
plt.plot(x, diff_first_angle[st:end])


# %%



def get_bendamps(tail_coords):
    n_rois = len(tail_coords[0]) 
    orients = np.empty((n_rois))
    orients[:] = np.nan

    bend_amps = np.empty((n_rois))
    bend_amps[:] = np.nan

    for fish in range(n_rois):
            x = tail_coords[0][fish, :]
            y = tail_coords[1][fish, :]
            x[x==0] = np.nan
            y[y==0] = np.nan

            angles = np.arctan2(np.diff(y), np.diff(x))
            
            angles = angles[~np.isnan(angles)]

            n_ang = len(angles)
            if n_ang > 1:
                orients[fish] = np.mean(angles)
                diff_angles = np.diff(angles)

                while np.max(np.abs(diff_angles)) > np.pi:
                    angles[np.where(diff_angles > np.pi)[0] + 1] = angles[np.where(diff_angles > np.pi)[0] + 1] - 2*np.pi
                    angles[np.where(diff_angles <-np.pi)[0] + 1] = angles[np.where(diff_angles <- np.pi)[0] + 1] + 2*np.pi
                    diff_angles = np.diff(angles)
                        
                
                bend_amps[fish] = np.sum(diff_angles)

    return orients, bend_amps