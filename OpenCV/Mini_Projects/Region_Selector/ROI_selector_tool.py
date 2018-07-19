"""
Select various ROI rectangles and save them
"""
import cv2 as cv
import time
import numpy as np

window_selected = False
sel_windows = {
    "num_windows": 0,
    "windows": {},
    "capturing_mode": False,
    "current_capture_point": None
}
def mouseInteraction(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        # Create a new window index and assign the Upper left corner to it
        sel_windows["windows"][sel_windows["num_windows"]] = {
            "UL": (x, y)
        }
        sel_windows["capturing_mode"] = True
        sel_windows["current_capture_point"] = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        # Assign this one the lower right corner
        sel_windows["windows"][sel_windows["num_windows"]]["LR"] = (x, y)
        # Verification of the corner
        (ul_x, ul_y) = sel_windows["windows"][sel_windows["num_windows"]]["UL"]
        (lr_x, lr_y) = sel_windows["windows"][sel_windows["num_windows"]]["LR"]

        if ul_x > lr_x or ul_y > lr_y: # Swap case
            sel_windows["windows"][sel_windows["num_windows"]]["UL"] = (min(ul_x, lr_x), min(ul_y, lr_y))
            sel_windows["windows"][sel_windows["num_windows"]]["LR"] = (max(ul_x, lr_x), max(ul_y, lr_y))

        sel_windows["num_windows"] += 1
        sel_windows["capturing_mode"] = False
    elif event == cv.EVENT_MOUSEMOVE:
        if sel_windows["capturing_mode"]:
            sel_windows["current_capture_point"] = (x, y)


NUM_CAMERA_ATTEMPTS = 5
RESPONSE_PAUSE = 0.5
CAMERA_CAPTURE_MODE = True
FILE_READING_MODE = not CAMERA_CAPTURE_MODE
CAM_NO = 0
cam = cv.VideoCapture(CAM_NO)
cv.namedWindow("Window Selection")
cv.setMouseCallback("Window Selection", mouseInteraction)
# Get a frame from the camera
frame = None
if CAMERA_CAPTURE_MODE:
    for i in range(0, NUM_CAMERA_ATTEMPTS):
        ret, frame = cam.read()
        if ret is True:
            break
        else:
            print("WARNING : {an}/{ta} Unable to connect to camera {cam_num}".format(
                an=i + 1, ta=NUM_CAMERA_ATTEMPTS, cam_num=CAM_NO
            ))
            time.sleep(RESPONSE_PAUSE)
            if i + 1 == NUM_CAMERA_ATTEMPTS:
                print("ERROR : Program terminated because camera {0} didn't respond".format(CAM_NO))
                exit(0)

while True:
    if CAMERA_CAPTURE_MODE:
        ret, frame = cam.read()
    disp_frame = frame.copy()
    # Draw rectangles on the disp_frame
    for i in range(sel_windows["num_windows"]):
        cv.rectangle(disp_frame, sel_windows["windows"][i]["UL"], sel_windows["windows"][i]["LR"], (0, 255, 0))
    if sel_windows["capturing_mode"]:
        cv.rectangle(disp_frame, sel_windows["windows"][sel_windows["num_windows"]]["UL"],
                     sel_windows["current_capture_point"], (255, 0, 0))
    cv.imshow("Window Selection", disp_frame)

    key = cv.waitKey(1) & 0xFF
    if key == 32 or key == ord('q'):   # Space key
        break
    elif key == ord('i'):   # Print the information
        # Info about selections
        print("INFO : You have {n} selections".format(n=sel_windows["num_windows"]))
        if sel_windows["num_windows"] > 0:
            print("INFO BLOCK : They are as follows")
            for i in range(sel_windows["num_windows"]):
                print(">> {n}) UL : {ul}, LR : {lr}".format(
                    n=i + 1,
                    ul=sel_windows["windows"][i]["UL"],
                    lr=sel_windows["windows"][i]["LR"]
                ))
        if sel_windows["capturing_mode"]:
            print("INFO : You are in capturing mode. Capturing region {r}, latest point captured {p}".format(
                p=sel_windows["current_capture_point"], r=sel_windows["num_windows"]
            ))

    elif key == ord('d'):   # Delete the most recently captured window
        if sel_windows["num_windows"] > 0:
            del sel_windows["windows"][sel_windows["num_windows"] - 1]
            sel_windows["num_windows"] -= 1
            print("INFO : Latest selection deleted ", end='')
        else:
            print("ERROR : No selections left to delete ", end='')
        print("you now have {n} selections left".format(n=sel_windows["num_windows"]))

cv.destroyAllWindows()
cam.release()
