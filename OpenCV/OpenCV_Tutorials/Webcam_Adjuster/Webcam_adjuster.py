"""
Made by : Avneesh Mishra

Purpose :
To check if the camera is working, make adjustments and to get feed information
from the camera

Working
Result : Camera calibration working perfectly
"""
import cv2 as cv
import numpy as np

# Application settings
cameraNumber = 0   # Camera number
verbose = False    # Echo all the changes

cam = cv.VideoCapture(cameraNumber)

# Generate camera adjuster for this property with ID as 'propertyID' named 'name'
def generate_cam_adjuster(propertyID, name=None):
    def adjust(to_long_value):
        val = to_long_value / 255
        cam.set(propId=propertyID, value=val)
        if verbose == True and name is not None:
            print("Property {propname} set to {value}".format(propname= name.lower(), value= val))
    return adjust


cv.namedWindow('Properties')
cv.createTrackbar('Contrast', 'Properties', int(cam.get(cv.CAP_PROP_CONTRAST) * 255), 255,
                  generate_cam_adjuster(cv.CAP_PROP_CONTRAST, name='contrast'))
cv.createTrackbar('Brightness', 'Properties', int(cam.get(cv.CAP_PROP_BRIGHTNESS) * 255), 255,
                  generate_cam_adjuster(cv.CAP_PROP_BRIGHTNESS, name='brightness'))
cv.createTrackbar('Saturation', 'Properties', int(cam.get(cv.CAP_PROP_SATURATION) * 255), 255,
                  generate_cam_adjuster(cv.CAP_PROP_SATURATION, name='saturation'))
cv.createTrackbar('Hue', 'Properties', int(cam.get(cv.CAP_PROP_HUE) * 255), 255,
                  generate_cam_adjuster(cv.CAP_PROP_HUE, name='hue'))
cv.createTrackbar('Gain', 'Properties', int(cam.get(cv.CAP_PROP_GAIN) * 255), 255,
                  generate_cam_adjuster(cv.CAP_PROP_GAIN, name='gain'))
# cv.createTrackbar('Exposure', 'Properties', int(cam.get(cv.CAP_PROP_EXPOSURE) * 255), 255,
#                   generate_cam_adjuster(cv.CAP_PROP_EXPOSURE, name='exposure'))

frame_mod_number = 0
while cam.isOpened():
    ret, frame = cam.read()
    # Comment/Uncomment if you want to flip
    frame = cv.flip(frame, 1)
    if not ret:
        continue
    else:   # The status reporter
        frame_mod_number += 1
        if frame_mod_number % 50 == 0 and verbose:
            print("Frame size is {shape}".format(shape=frame.shape))
            frame_mod_number = 0
    cv.imshow('Fame', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cv.destroyAllWindows()
cam.release()
