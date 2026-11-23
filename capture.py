import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
cur_time = time.perf_counter()
N = 300
dtime_values = np.zeros(N)
frame_count = 0
filled = False

def display_fps(frame, loc):
    global cur_time, frame_count, filled
    prev_time = cur_time
    cur_time = time.perf_counter()
    d_time = cur_time-prev_time
    dtime_values[frame_count] = d_time
    frame_count += 1
    if frame_count > N-1:
        filled = True
        frame_count = 0
    if filled:
        avg_dtime = dtime_values.mean()
    else:
        avg_dtime = dtime_values[0:frame_count].mean()
    avg_fps = 1/avg_dtime
    cv2.putText(frame, f"Avg fps: {avg_fps:.1f}", loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1)
    return 0

def find_brightest_spot_opencv(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (minValue_bright, maxValue_bright, minLoc_bright, maxLoc_bright) = cv2.minMaxLoc(gray)
    return maxLoc_bright

def find_brightest_spot_2forLoops(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightest_spot = [0,0,0] # [value, colum, row]
    for row in range(gray.shape[0]):
        for colum in range(gray.shape[1]):
            value = gray[row, colum]
            if value > brightest_spot[0]:
                brightest_spot = [value, colum, row]
    return tuple(brightest_spot[1:3])

def find_brightest_spot_numpy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightest_spot_count = np.argmax(gray) # position of the max value, returns int as position if it were 1 long row
    row, col = np.unravel_index(brightest_spot_count, gray.shape)    # turn into (row, column)
    return (col, row)

def find_reddest_bgr(frame):
    blue = frame[:,:,0]
    green = frame[:,:,1].astype(np.float32) # needed for the blue + green operation to prevent overflow
    red = frame[:,:,2].astype(np.float32) # needed for the blue + green operation to prevent overflow

    reddest_bgr = red-(blue + green)/2
    (minValue_red_bgr, maxValue_red_bgr, minLoc_red_bgr, maxLoc_red_bgr) = cv2.minMaxLoc(reddest_bgr)
    return maxLoc_red_bgr

def find_reddest_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].astype(np.float32)
    hue[(hsv[:,:,1] < 100) | (hsv[:,:,2] < 50)] = 90 # excluding white to black colors since they are given hue 0

    (minValue_red_hsv, maxValue_red_hsv, minLoc_red_hsv, maxLoc_red_hsv) = cv2.minMaxLoc(hue)
    if minValue_red_hsv < 180-maxValue_red_hsv: # 180° is same as 0° thats why 180-maxValue... because 0° is red-der than 179°
        reddest_hsv_loc = minLoc_red_hsv
    else:
        reddest_hsv_loc = maxLoc_red_hsv
    return reddest_hsv_loc

brightest_spot_time = 0
iteration = 0

Show_display = True

run_sec = 10
num_frames = 0
start_run = time.perf_counter()


while(True):
    ret, frame = cap.read()

    #start_time = time.perf_counter()

    brightest_loc_opencv = find_brightest_spot_opencv(frame)

    #brightest_loc_2forLoops = find_brightest_spot_2forLoops(frame)

    #brightest_loc_numpy = find_brightest_spot_numpy(frame)

    #end_time = time.perf_counter()
    #brightest_spot_time += end_time-start_time
    #iteration += 1
    
    reddest_bgr_loc = find_reddest_bgr(frame)

    reddest_hsv_loc = find_reddest_hsv(frame)
    
    if Show_display:
        # Draw circles:
        thickness = 2
        cv2.circle(frame, brightest_loc_opencv,15,(0,0,0),thickness) # brightest spot
        #cv2.circle(frame, brightest_loc_2forLoops, 15, (0,0,0), thickness)
        #cv2.circle(frame, brightest_loc_numpy, 15, (0,0,0), thickness)
        cv2.circle(frame, reddest_bgr_loc, 15, (0,255,0),thickness) # reddest spot BGR appears as a green circle
        cv2.circle(frame, reddest_hsv_loc, 15, (0,0,255),thickness) # reddest spot HSV appears as a red circle

        display_fps(frame, (10, 30))

        cv2.imshow('frame',frame)
    else:
        num_frames += 1
        if num_frames == 150: # 5 frames at 30fps so start up doesn't change timing
            start_run = time.perf_counter()
        if time.perf_counter() - start_run > run_sec:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Average time to find brightest spot took: {brightest_spot_time*1000/iteration:.4f} ms, taken over {iteration} iterations")
if not Show_display:
    print(f"Average fps without display is {(num_frames-150)/run_sec:.1f}")


