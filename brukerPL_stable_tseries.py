#%%

import win32com.client as client
import time
import matplotlib.pyplot as plt
import cv2
import napari
import PySimpleGUI as sg
pl = client.Dispatch("PrairieLink64.Application")
import numpy as np

import glob, os, tqdm
from natsort import natsorted
from PIL import Image
from scipy.ndimage import zoom

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure


filename = '20220831_HuCGCaMP7f_6dpf_CGP7930_fish2'
func_tseries_name = 'func_series_3hr.env'

t_series_dir = os.path.realpath(r'C:\Users\User\Desktop\Randlett_Z-Drive\t_series')

reg_downsample = 0.5 # proportion to downsample image before registration
#
pl.Connect()
#%

# pl.SendScriptCommands('-DoNotWaitForScans')
# pl.SendScriptCommands('-LimitGSDMABufferSize false')
# pl.SendScriptCommands('-srd true 128')
pl.SendScriptCommands('-fa 1');  
pl.SendScriptCommands("-SetAcquisitionMode ResonantGalvo")



samplesPerPixel      = pl.SamplesPerPixel()
pixelsPerLine        = pl.PixelsPerLine()
linesPerFrame        = pl.LinesPerFrame()
totalSamplesPerFrame = samplesPerPixel*pixelsPerLine*linesPerFrame
samples_buffer = np.zeros(totalSamplesPerFrame*10, dtype='uint16')
#%

def setup_zstack(position_start, position_end, step_size, save_name, z_index = 1):
# z_index =1#0=normal z drive, 1=piezo . might need to have z-focus selected in PrairieView

    if pl.GetState('ZDevice') == '0':
        pl.SendScriptCommands('-sts ZDevice 1'); 
    #%
    pl.SendScriptCommands("-ma Z %s %s" %  (position_start, z_index))
    time.sleep(0.5)
    pl.SendScriptCommands('-zsb onlyMotors false false')
    pl.SendScriptCommands("-ma Z %s %s" %  (position_end, z_index))
    time.sleep(0.5)
    pl.SendScriptCommands('-zse onlyMotors false false')
    pl.SendScriptCommands('-zsz %s' % (step_size))
    pl.SendScriptCommands('-zsmt start') 

    pl.SendScriptCommands('-zss %s' %  (save_name) )

# set up anatomy stack

pl.SendScriptCommands('-zsd All')
pl.SendScriptCommands('-fa 32'); 

anat_series = os.path.join(t_series_dir, 'anat_series.env')
pl.SendScriptCommands('-tsl ' + anat_series)
setup_zstack(180, 320, 1, 'AnatStack')

pl.SendScriptCommands('-SetSavePath H:/')
pl.SendScriptCommands('-fn Tseries %s' %  (filename + '_anatomy'))
pl.SendScriptCommands('-fi Tseries 0')

print('Adjust anatomy stack using the z-drive so that the piezo range covers the correct areas, then run next cell')
#%% start anatomy stack tSeries

pl.SendScriptCommands('-TSeries')
print('..................')
print('when stack is done being acquired (~10 min), use the block ripping utility to convert the anatomy stack to tiff files, then run next cell')
#%%

# load anatomy stack for reference when stabilizing stack. Set up functional t-series and start it

folder = glob.glob('H:/'+filename+'_anatomy*')[-1]

tiffs = natsorted(glob.glob(folder + '/*.tif'))

n_slices_anat = 141
anat_stack = np.zeros((n_slices_anat, linesPerFrame, pixelsPerLine), dtype=float)

for i in tqdm.tqdm((range(len(tiffs)))):
    sl = Image.open(tiffs[i])
    anat_stack[i%n_slices_anat, :, :] = anat_stack[i%n_slices_anat, :, :] + sl
plt.figure(figsize=(40,40))
plt.imshow(np.max(anat_stack, axis=0))
plt.show()


anat_stack_ds = zoom(anat_stack, [1, reg_downsample,reg_downsample])


#%
z_series_start = 200
z_series_end = 310
pl.SendScriptCommands('-fa 1'); 
setup_zstack(z_series_start, z_series_end, 10, 'FuncStack')
func_series = os.path.join(t_series_dir, func_tseries_name)
pl.SendScriptCommands('-tsl ' + func_series)
pl.SendScriptCommands('-SetSavePath H:/')
pl.SendScriptCommands('-fn Tseries %s' %  (filename + '_func'))
pl.SendScriptCommands('-fi Tseries 0')
pl.SendScriptCommands('-zsd Middle')
time.sleep(5)
pl.SendScriptCommands('-TSeries')
print('.................................')
print('Wait until functional stack starts acquiring. Then run the next cell')
#%
#%%
# run stabilization routine. 

from skimage.registration import phase_cross_correlation

def find_best_plane(im, anat_stack, plot=True):
    errors_blocks = np.zeros(n_slices_anat)
    shifts_blocks = np.zeros((n_slices_anat, 2))
    for slice in range(n_slices_anat):
        shifts_blocks[slice, :], errors_blocks[slice], phasediff = phase_cross_correlation(anat_stack[slice,:,:],im, normalization=None)
    best_slice = np.argmin(errors_blocks)
    if plot:
        plt.plot(errors_blocks)
        plt.title(best_slice)
        plt.show()
        print('best plane = ' + str(best_slice))
    return best_slice, shifts_blocks[best_slice, :], errors_blocks

if pl.GetState('ZDevice') == '1':
    pl.SendScriptCommands('-sts ZDevice 0');  

st = time.time()
im = np.zeros((linesPerFrame, pixelsPerLine), dtype=float)

while time.time() - st < 30:
    im = im + np.array(pl.GetImage_2(2,linesPerFrame,  pixelsPerLine))


plt.imshow(im)
plt.show()
original_slice, shifts, errors = find_best_plane(zoom(im, [reg_downsample,reg_downsample]), anat_stack_ds, plot=True)
im_orig = im.copy()
im_orig = im_orig/np.max(im_orig)
im[:] = 0
#%
px_sz = 0.58
max_z = 50
max_xy = 200
refresh_seconds = 30

overlay_im = sg.Image(filename='', key='overlay_im')
stack_im = sg.Image(filename='', key='stack_im')
anat_stack_8bit = anat_stack.copy()
anat_stack_8bit = 255*anat_stack_8bit/np.max(anat_stack_8bit)
anat_stack_8bit = anat_stack_8bit.astype('uint8')
best_slice = original_slice.copy()
layout = [
    [sg.Text(filename, key = 'filename', size=(50,1))],
    [overlay_im, stack_im],
    [
        sg.Canvas(key='figCanvas'), 
        [   
            sg.Text('refresh seconds', size=(10,1)),
            sg.InputText(default_text = str(refresh_seconds), key = 'refresh', size=(10,1))
        ],
        [   
            sg.Text('time until refresh: = ', size=(10,1)),
            sg.Text(str(refresh_seconds), key = 'ref_time', size=(10,1)),
        ]
    ]

    ]
#%
window = sg.Window(
    'Drift_correction.py', 
    layout, 
    location=(0, 0), 
    return_keyboard_events=True, 
    use_default_focus=False)
window.Finalize()
window.TKroot.focus_force()

canvas_elem = window['figCanvas']
canvas = canvas_elem.TKCanvas
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


fig = Figure()
ax = fig.gca()
ax.plot(errors)
ax.vlines(original_slice, 0.2, 0.7, 'k')
fig_agg = draw_figure(canvas, fig)


def adjust_position(best_slice, shifts):
    delta = best_slice - original_slice
    if abs(delta) >= 1 and abs(delta) < max_z:
        print('moving in Z by ' + str(delta))
        pos = pl.GetMotorPosition("Z", 0) # currently this only seems to work for whatever the active one in the software is
        print('Z motor position = ' + str(pos))
        position = pos - delta
        pl.SendScriptCommands("-ma Z %s %s" %  (position, 0))

    elif abs(delta) >= max_z:
        print('WARNING... shfit to big in z,  not performing')
        print(delta)

    if abs(shifts[0]) >= 1 and abs(shifts[0]) < max_xy:
        y_shift = shifts[0] * px_sz / reg_downsample
        print('moving in Y by ' + str(y_shift))
        y_pos = pl.GetMotorPosition('Y')
        y_position = y_pos - y_shift
        pl.SendScriptCommands("-ma Y %s" %  (y_position))

    elif abs(shifts[0]) >= max_xy:
        print('WARNING... shfit to big in Y,  not performing')
        print(shifts[0])

    if abs(shifts[1]) >= 1 and abs(shifts[1]) < max_xy:
        x_shift = shifts[1] * px_sz / reg_downsample
        print('moving in X by ' + str(x_shift))
        x_pos = pl.GetMotorPosition('X')
        x_position = x_pos + x_shift
        pl.SendScriptCommands("-ma X %s" %  (x_position))
    
    elif abs(shifts[1]) >= max_xy:
        print('WARNING... shfit to big in X,  not performing')
        print(shifts[1])
#%
st = time.time()


while True:
    
    while time.time() - st < refresh_seconds:
        im = im + np.array(pl.GetImage_2(2,linesPerFrame,  pixelsPerLine))
        event_im, values_im = window.Read(timeout=1, timeout_key='timeout')
        window['ref_time'].update(value = str(np.round(refresh_seconds - (time.time() - st))))
        if event_im is None:
            break

    if event_im is None:
        break
    
    im = im/np.max(im)
    im_disp = np.stack((im_orig, im, im_orig), axis=2)
    im_disp = im_disp*255
    im_disp = im_disp.astype('uint8') * 2

    stack_disp = np.stack((anat_stack_8bit[original_slice, :, :], im*255 , anat_stack_8bit[original_slice, :, :]), axis=2) 
    stack_disp = stack_disp * 2
    overlay_im.Update(data=cv2.imencode('.png', im_disp)[1].tobytes())  
    stack_im.Update(data=cv2.imencode('.png',  stack_disp)[1].tobytes())
    


    best_slice, shifts, errors = find_best_plane(zoom(im, [reg_downsample,reg_downsample]), anat_stack_ds, plot=False)
    ax.cla()
    ax.plot(errors)
    ax.vlines(original_slice, 0.2, 0.7, 'k')
    fig_agg.draw()


    adjust_position(best_slice, shifts)
    

    if event_im is None:
        break

    refresh_seconds = int(values_im['refresh'])
    im[:] = 0
    st = time.time()
    
window.close()

