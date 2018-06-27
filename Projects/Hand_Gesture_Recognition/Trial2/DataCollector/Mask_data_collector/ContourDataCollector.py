"""
Created by Avneesh on 18/6/18
"""
import cv2 as cv
import numpy as np

cam_no = 0
cam = cv.VideoCapture(cam_no)
# Your declaration or starting code here
bgsub_nolearn = cv.createBackgroundSubtractorMOG2()
bgsub_learning = cv.createBackgroundSubtractorMOG2()

# Buffer
for i in range(20):
    _, img = cam.read()
    _ = bgsub_learning.apply(img)
    _ = bgsub_nolearn.apply(img)

learning_mask = False
mask_learned = False
upper_threshold = 50  # Threshold for detection of moving objects
lower_threshold = 10

rect_shape = (190, 235)             # width, height
rect_center_exact = (140, 235.5)    # c_x, c_y
rect_center = (int(rect_center_exact[0]), int(rect_center_exact[1]))
rect_pt1 = (int(rect_center_exact[0] - rect_shape[0]/2),
            int(rect_center_exact[1] - rect_shape[1]/2))
rect_pt2 = (int(rect_center_exact[0] + rect_shape[0]/2),
            int(rect_center_exact[1] + rect_shape[1]/2))
# A 235 rows by 190 column rectangle

print("Initialised camera {cam_number}.\n".format(cam_number= cam_no),
      "Capturing frames of shape {frame_shape} and {frame_num_channels} channels\n".format(
          frame_shape=tuple(reversed(img.shape[:2])), frame_num_channels=img.shape[2]),
      "Recording rectangle is {p1}, {p2}. Shape of images are ({shapeX}, {shapeY})\n".format(
          p1=rect_pt1, p2=rect_pt2, shapeX=rect_pt2[0]-rect_pt1[0],
          shapeY=rect_pt2[1]-rect_pt1[1]),
      "Bounding limits are X:({x_min}, {x_max}), Y:({y_min}, {y_max})".format(
          x_min= rect_shape[0]/2, x_max= img.shape[1] - rect_shape[0]/2,
          y_min= rect_shape[1]/2, y_max= img.shape[0] - rect_shape[1]/2),
      sep=''
      )

pass

# To save the images into a new folder
def save_mask_learned():
    # Save the image
    try:
        with open('Index.txt', 'r+') as fin:
            number = int(fin.read())
    except:
        with open('Index.txt', 'w+') as fout:
            fout.write('1')
            number = 0
    else:
        with open('Index.txt', 'w+') as fout:
            fout.write(str(number + 1))
    print("Saving files with file number {num} in Data folder".format(num=number))
    cv.imwrite('Data/rect_frame{index}.jpg'.format(index=number), rect_frame)
    cv.imwrite('Data/mask_no_learn{index}.jpg'.format(index=number), mask_hand_to_save)


# To reset the masks in the rectangle
def reset_learned_masks():
    global bgsub_learning, bgsub_nolearn, mask_learned, learning_mask
    cv.destroyAllWindows()
    learning_mask = False
    mask_learned = False
    bgsub_nolearn = cv.createBackgroundSubtractorMOG2()
    bgsub_learning = cv.createBackgroundSubtractorMOG2()

while cam.isOpened():
    # Capture frame and retry if frame dropped
    ret, frame = cam.read()
    if not ret:
        continue

    # Your loop code here
    frame = cv.flip(frame, 1)
    frame = cv.GaussianBlur(frame, (5,5), 3)
    # cv.imshow("Live frame", frame)
    frame_disp = frame.copy()
    if mask_learned:
        rect_color = (0, 255, 0)
    else:
        rect_color = (0, 0, 255)
    cv.rectangle(frame_disp, rect_pt1, rect_pt2, rect_color)
    # Masks
    mask_nolearn = bgsub_nolearn.apply(frame[rect_pt1[1]:rect_pt2[1], rect_pt1[0]:rect_pt2[0], :],
                                       learningRate= 0.001)
    _, mask_nolearn = cv.threshold(mask_nolearn, 130, 255, cv.THRESH_BINARY)
    mask_nolearn = cv.erode(mask_nolearn, np.ones((3, 3), dtype= np.uint8))
    mask_nolearn_disp = mask_nolearn.copy()
    mask_nolearn_disp = cv.cvtColor(mask_nolearn_disp, cv.COLOR_GRAY2BGR)
    avg_value_no_learn = np.average(mask_nolearn)

    mask_learning = bgsub_learning.apply(frame[rect_pt1[1]:rect_pt2[1], rect_pt1[0]:rect_pt2[0], :],
                                         learningRate = 0.01)
    _, mask_learning = cv.threshold(mask_learning, 130, 255, cv.THRESH_BINARY)
    mask_learning = cv.erode(mask_learning, np.ones((3,3), dtype= np.uint8))
    mask_learning_disp = mask_learning.copy()
    mask_learning_disp = cv.cvtColor(mask_learning_disp, cv.COLOR_GRAY2BGR)
    avg_value_learning = np.average(mask_learning)

    cv.putText(mask_nolearn_disp, "{avg}".format(avg=avg_value_no_learn),
               (20, 220), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
    cv.putText(mask_learning_disp, "{avg}".format(avg=avg_value_learning),
               (20, 220), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))

    cv.imshow("Mask (no learning)", mask_nolearn_disp)
    cv.imshow("Mask (learning)", mask_learning_disp)

    # Activate mask learning mode
    if avg_value_learning > lower_threshold and avg_value_no_learn > upper_threshold\
            and (not learning_mask) and (not mask_learned):
        learning_mask = True
    if learning_mask:
        cv.putText(frame_disp, "Learning Mask", (40, 470), cv.FONT_HERSHEY_COMPLEX,
                   0.7, (0, 255, 0))
    # Wait for average of faster learning mask to go down
    if avg_value_no_learn > upper_threshold and avg_value_learning < lower_threshold\
            and learning_mask:
        learning_mask = False
        mask_learned = True
        rect_frame = frame[rect_pt1[1]:rect_pt2[1], rect_pt1[0]:rect_pt2[0], :]
        rect_frame[mask_nolearn != 255, :] = 0
        rect_frame_hsv = cv.cvtColor(rect_frame, cv.COLOR_BGR2HSV)
        rect_frame_h = rect_frame_hsv[:,:,0]
        rect_frame_h[mask_nolearn != 255] = 0
        rect_frame_s = rect_frame_hsv[:,:,1]
        rect_frame_s[mask_nolearn != 255] = 0
        rect_frame_v = rect_frame_hsv[:,:,2]
        rect_frame_v[mask_nolearn != 255] = 0
        mask_hand_to_save = mask_nolearn
        cv.imshow("Hue", rect_frame_h)
        cv.imshow("Saturation", rect_frame_s)
        cv.imshow("Variance", rect_frame_v)
        cv.imshow("Rect Frame", rect_frame)
        cv.imshow("Scanning Mask", mask_nolearn)
        max_H = np.max(rect_frame_h[mask_nolearn == 255])
        min_H = np.min(rect_frame_h[mask_nolearn == 255])
        max_S = np.max(rect_frame_s[mask_nolearn == 255])
        min_S = np.min(rect_frame_s[mask_nolearn == 255])
        max_V = np.max(rect_frame_v[mask_nolearn == 255])
        min_V = np.min(rect_frame_v[mask_nolearn == 255])

    if mask_learned:
        cv.putText(frame_disp,
                   "Max HSV : {maxH}, {maxS}, {maxV}".format(
                       maxH= max_H, maxS= max_S, maxV= max_V
                   ), (50, 440), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255))
        cv.putText(frame_disp,
                   "Min HSV: {minH}, {minS}, {minV}".format(
                       minH=min_H, minS=min_S, minV=min_V
                   ), (50, 470), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 255))

    cv.imshow("Frame", frame_disp)
    cv.moveWindow("Frame", 100, 100)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        if key == 27:
            # Emergency escape, don't learn data
            mask_learned = False
        break
    elif key == ord('p'):
        while True:
            key = cv.waitKey(0) & 0xff
            if key == ord('p'):
                break
            elif key == ord('q') or key == 27:
                cv.destroyAllWindows()
                cam.release()
                exit(0)
    elif key == ord('r'):
        reset_learned_masks()
    elif key == ord('s'):
        save_mask_learned()
        reset_learned_masks()

# Your end code here
if mask_learned and cv.waitKey(0) & 0xff == ord('s'):
    # Save the mask learned
    save_mask_learned()

cv.destroyAllWindows()
cam.release()