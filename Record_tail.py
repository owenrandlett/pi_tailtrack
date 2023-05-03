import picamera, time, cv2
import numpy as np
from PIL import Image
from datetime import datetime
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

stim_pin = 4 # pin that is listening to the stimulus line from the microscope, used for syncing between raspberry pi and microscope
stim_val = 1 # will be used to store the value of the stimulus line

GPIO.setup(stim_pin, GPIO.IN)
RECORD_TIME = 20000000# number of seconds to record, this can be used to end the experiment or you can just close the window. 

# tracking parameters for adaptive thesholding and morphology operations. Can be adjusted to modify tracking based on indifvidual lighting, etc. 
thresh = -10
n_hood = 33

structuringElement_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
structuringElement_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

RECORD_DATA = True
RECORD_VIDEO = False
date_str = datetime.now().strftime("%Y%m%d--%H%M%S")
print(date_str)

if RECORD_DATA:
	f_coords = open('/home/pi/Desktop/Data/'+date_str+'_coords.txt','w')
	f_tstamps = open('/home/pi/Desktop/Data/'+date_str+'_tstamps.txt','w')


SENSOR_MODE = 0  
RESOLUTION = (128,128)
FRAME_RATE = 120


FPS_MODE_OFF = 0
FPS_MODE_T0 = 1
FPS_MODE_FBF = 2
FPS_MODE = 1

# Calculate the actual image size in the stream (accounting for rounding
# of the resolution)
# Capturing yuv will round horizontal resolution to 16 multiple and vertical to 32 multiple
# see: https://picamera.readthedocs.io/en/release-1.13/recipes2.html#unencoded-image-capture-yuv-format
fwidth = (RESOLUTION[0] + 31) // 32 * 32
fheight = (RESOLUTION[1] + 15) // 16 * 16
print(f'frame size {fwidth}x{fheight}')

if RECORD_VIDEO:
	out= cv2.VideoWriter('/home/pi/Desktop/Data/'+date_str+'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (fwidth,fheight))
 

frame_cnt = 0 # counter for frames

# physical dimensions of fish and tail segements for tracking
tail_len = 70
n_seg = 10
seg_len = tail_len/n_seg
pts_x = np.zeros(n_seg+1)
pts_y = np.zeros(n_seg+1)

search_degrees = np.pi/2

start_coords = np.array((50, fheight/2)) # coordinate for the base of the tail is hard coded, will search to the right of this for tail in the semented image. Fish will need to be moved properly into position, with tail poiting to the right

kill_recording = False
t0 = None
t_prev = None
fish_mask = np.zeros(RESOLUTION, dtype=bool)

# class for handling the video stream
# code for dealing with the camera is based on https://raspberrypi.stackexchange.com/questions/58871/pi-camera-v2-fast-full-sensor-capture-mode-with-downsampling/58941#58941

class show_output(object): 
	def write(self, buf): # video data is appended to this method, will be called on every frame

		# write will be called once for each frame of output. buf is a bytes
		# object containing the frame data in YUV420 format; we can construct a
		# numpy array on top of the Y plane of this data quite easily:

		global frame_cnt, t_prev, stim_val, kill_recording, thresh, n_hood

		stim_val = GPIO.input(stim_pin) # check the stimulus pin and record value (high or low)
		y_data = np.frombuffer(buf, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth)) # get image frame

		pts_x[:] = np.nan
		pts_y[:] = np.nan

		# use adaptive thresholding to identify the fish tail. Parameters will be critical here and can be experimented with
		th = cv2.adaptiveThreshold(
			y_data,
			255,
			cv2.ADAPTIVE_THRESH_MEAN_C,
			cv2.THRESH_BINARY,
			n_hood,
			thresh
			)

		# clean up the thresholded blobs with an opening to remove small objects and then a closing operation to fuse remaining objects that are nearly touching. Aggressiveness of this may need to be modified. 
		th2 = cv2.morphologyEx(
			cv2.morphologyEx(th, cv2.MORPH_OPEN, structuringElement_open),
			cv2.MORPH_CLOSE,
			structuringElement_close)
		
		found_stuff = np.sum(th2) > 0 # any blobs left? 
		
		fish_mask = th2 > 0 # get a binary image that will (hopefully) mostly represent the fish after thresholding and smoothing

		if found_stuff:

			pts_x[0], pts_y[0] = start_coords
			
			for i in range(n_seg):
				if i == 0:
					search_angle = np.arctan2(
								0, 
								pts_x[0]
							   )
					
				else:
					search_angle = np.arctan2(
                        pts_y[i]-pts_y[i-1], 
                        pts_x[i]-pts_x[i-1]
                        )
					
				ang = np.arange(search_angle-search_degrees,search_angle+search_degrees,np.pi/30)
				
				ArcCoors =(pts_y[i] + np.sin(ang)*seg_len).astype(np.int64), (pts_x[i] + np.cos(ang)*seg_len).astype(np.int64)
				
				indKeep = np.logical_and(
					np.logical_and(ArcCoors[0] >=0 ,  ArcCoors[0] <= fheight-1) ,
					np.logical_and(ArcCoors[1] >=0 ,  ArcCoors[1] <= fwidth-1) 
					)

				ArcCoors = (ArcCoors[0][indKeep], ArcCoors[1][indKeep])
				
				ind_in_mask = fish_mask[ArcCoors[0], ArcCoors[1]]
				#print(ind_in_mask)
				pts_x[i+1] = np.mean(ArcCoors[1][ind_in_mask])
				pts_y[i+1] = np.mean(ArcCoors[0][ind_in_mask])
				n_good = np.sum(ind_in_mask)
				if n_good == 0: # if we found no valid points break the loop
					break

		dt = time.time() - t0  # dt
		fps = frame_cnt / dt
		
		if RECORD_DATA:
			np.savetxt(f_coords, (pts_x, pts_y), fmt='%10.1f', delimiter=',')
			np.savetxt(f_tstamps, [dt, stim_val], fmt='%10.3f', delimiter=',')

		# display image and tracking information, not every frame because rendering is slow
		if frame_cnt % 20 == 0 and frame_cnt>0:
			
			raw_im = np.copy(y_data)
			if found_stuff:
				raw_im[pts_y.astype(int),pts_x.astype(int)] = 255
				raw_im[ArcCoors[0], ArcCoors[1]] = 255
			im_disp = cv2.resize(
				cv2.hconcat([raw_im, 255-th.astype('uint8'), 255*(1-fish_mask.astype('uint8'))]),
				[RESOLUTION[0]*4*3, RESOLUTION[1]*4]
				)
			im_disp[:70, : ] = im_disp[:70, : ] * 0.2
			cv2.putText(im_disp,
						f'frame = {frame_cnt} ; sec_elapsed = {int(round(dt, 0))} ; fps = {round(fps, 2)} ; stim_val = {round(stim_val, 0)}',
						(5,50), cv2.FONT_HERSHEY_SIMPLEX,
						1.2,
						200,
						thickness=4)
			cv2.imshow('frame', im_disp)
			key = cv2.waitKey(1) & 0xFF
			
			
			if key == 84:
				start_coords[1] += 1
			elif key == 82:
				start_coords[1] -= 1
			elif key == 81:
				start_coords[0] -= 1
			elif key == 83:
				start_coords[0] += 1

			elif key == 119: # press 'w' to increase threshold
				thresh+=1
				print('threshold = ' + str(thresh))
			elif key == 115: # press 's' to increase threshold
				thresh-=1
				print('threshold = ' + str(thresh))
			elif key == 100: # press 'w' to increase threshold
				n_hood += 2
				print('neightborhood = ' + str(n_hood))
			elif key == 97: # press 'w' to increase threshold
				n_hood -= 2
				print('neightborhood = ' + str(n_hood))
			elif key == 113: # press 'q' to quit
				kill_recording = True
			
			#print(key)
			if RECORD_VIDEO:
				out.write(im_disp)
		frame_cnt += 1

	def flush(self):
		pass  # called at end of recording

with picamera.PiCamera(
        sensor_mode=SENSOR_MODE,
        resolution=RESOLUTION,
        framerate=FRAME_RATE
	) as camera:
    print('camera setup')
	
	# camera settings
    camera.rotation = 0
    camera.hflip = True
    camera.iso = 800
    camera.exposure_mode = 'auto'
    camera.awb_mode = 'off'
    camera.awb_gains = 1
    camera.image_denoise = False

    time.sleep(0.1)  # let the camera warm up and set gain/white balance

    print('starting recording')
    output = show_output()

    t0 = time.time()  # seconds
    t_prev = t0

    camera.start_recording(output, 'yuv')
    while (time.time() - t0) < RECORD_TIME and not kill_recording:
        camera.wait_recording(0.001)
    camera.stop_recording()

    dt = time.time() - t_prev
    print(f'total frames: {frame_cnt}')
    print(f'time recording: {round(dt, 2)}s')
    fps = frame_cnt / dt
    print(f'fps: {round(fps, 2)}s')
cv2.destroyAllWindows()
f_coords.close()
f_tstamps.close()

print('done')

