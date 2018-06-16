"""
Created by stark on 15/6/18

Mask made with a background subtraction operation (with 0 learning rate)
Use a mask (HSV) to get skin only
Detect biggest contour after removing all the faces and then do contour analysis on it

NOTE :
Background must be stationary for this, so this can't be used for cameras in motion with so much accuracy

Result : Kind of fine

"""
import cv2 as cv
import numpy as np

cam_no = 0
cam = cv.VideoCapture(cam_no)
# Your declaration or starting code here
bgsub = cv.createBackgroundSubtractorMOG2()
# Comment while execution (required only for intellisence
# bgsub = cv.BackgroundSubtractorMOG2(bgsub)
learning_rate = 0

cv.namedWindow('Threshold Properties')
cv.createTrackbar('Contour Distance', 'Threshold Properties', 1000, 20000, lambda x: None)
# 6842 was the best value while testing
cv.setTrackbarPos('Contour Distance', 'Threshold Properties', 6842)
cv.createTrackbar('Area', 'Threshold Properties', 20, 20000, lambda x: None)
# 3383 was a good value while testing
cv.setTrackbarPos('Area', 'Threshold Properties', 3383)


# View frame information
def view_channels(img_frame):
    # Show BGR channel
    cv.imshow("Blue", img_frame[:, :, 0])
    cv.imshow("Green", img_frame[:, :, 1])
    cv.imshow("Red", img_frame[:, :, 2])
    # Show HSV channel
    frame_hsv = cv.cvtColor(img_frame, cv.COLOR_BGR2HSV)
    cv.imshow("Hue", frame_hsv[:, :, 0])
    cv.imshow("Saturation", frame_hsv[:, :, 1])
    cv.imshow("Variance", frame_hsv[:, :, 2])


def getBoundsHSV_HandColors():
    """
    :return:
        Get the HSV bounds for the hand skin tone extraction
    """
    upperb = (1, 65, 95)
    lowerb = (12, 200, 205)
    return upperb, lowerb


# To return the mask of things that aren't skin (255 - No skin, 128 - Maybe skin, 0 - don't know
def not_skin_mask(img, removeFace = False):
    """
        Generates a mask of things that are not remotely close to skin colour
        The result has 255 for every point that can safely be removed and 0 for every point that must not be removed
        It returns 128 for maybe skin
    :param img:
        Image to process
    :return:
        mask
    """
    # Greater red pixels will have 255
    mask_greater_red = np.ones_like(img[:, :, 0], dtype=np.uint8) * 255
    mask_greater_red[img[:, :, 0] < img[:, :, 1]] = 0
    mask_greater_red[img[:, :, 0] < img[:, :, 2]] = 0
    # mask_high_variance = np.zeros_like(img, )
    mask_maybe_skin = np.zeros_like(img[:, :, 0], dtype=np.uint8)
    # Got from testing
    lower_hsv_bound, higher_hsv_bound = getBoundsHSV_HandColors()
    mask_skin = cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), lower_hsv_bound, higher_hsv_bound)
    mask_maybe_skin[mask_skin == 255] = 128

    # TODO - Make something to remove things that are stationary in the frame
    # You can use the stationary_object_mask
    # TODO - Remove the face from the region if removeFace is True
    # Use a Trained Haarcascade classifier

    mask = mask_greater_red
    mask[mask_maybe_skin == 128] = 128
    return mask


# To return the mask of objects that are stationary (255 - Stationary, 128 - Maybe in motion, 0 - definitely in motion)
def stationary_object_mask(img):
    """
        Generates a mask of objects that are stationary in the scene. For a completely stationary object, value is 255
        For a completely stationary object, value is 0. 128 for maybe stationary
    :param img: input image
    :return:
        mask
    """
    global learning_rate
    mask = bgsub.apply(img, learningRate=learning_rate)
    mask = 255 - mask  # Toggle everything
    return mask


# Perform contour analysis
def contour_analysis(retInfo=False, skin_mask = None):
    """
        Make contours and analyze the frame based on all information known, return a dictionary if needed
    :param retInfo:
        Return detailed dictionary or not
    :parameter skin_mask:
        This mask must be defined (as a variable named 'mask_skin') before if not passed
    :return disp_frame:
        The detailed analysed frame
    :return ret_info:
        A dictionary with structure
        "result"
            "hands"
                index_number
                    "hand_number" : number of hands,
                    "bounding_rect" : bounding rectangle of the hand (x,y,w,h) format,
                    "number_fingers" : number of fingers shown
            "number_hands" : number of hands found in the image

    """
    if skin_mask is None:
        skin_mask = mask_skin
    # Contour analysis
    _, contours, hierarchy = cv.findContours(skin_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    disp_frame = frame.copy()
    number_hands = 0
    ret_info = {}
    ret_info["result"] = {}
    ret_info["result"]["hands"] = {}
    # Check each and every contour
    for i in range(len(contours)):
        cnt = contours[i]
        [_, _, _, parent] = hierarchy[0, i]
        # Assume hand is the visible contour and the parent
        if parent != -1:
            continue
        area = cv.contourArea(cnt)
        # Area threshold
        if area < cv.getTrackbarPos('Area', 'Threshold Properties'):
            continue
        hull = cv.convexHull(cnt, returnPoints=True)
        cv.drawContours(disp_frame, [cnt], 0, (255, 0, 0))
        cv.drawContours(disp_frame, [hull], 0, (0, 255, 0))
        hull = cv.convexHull(cnt, returnPoints=False)
        convexity_defects = cv.convexityDefects(cnt, hull)
        # No convexity defects
        if convexity_defects is None:
            continue
        # Iterate over defects (to get gesture)
        gesture_number = 0
        for j in range(len(convexity_defects)):
            defect = convexity_defects[j]
            (s, e, f, distance) = defect[0]
            if distance < cv.getTrackbarPos('Contour Distance', 'Threshold Properties'):
                continue
            start = tuple(cnt[s, 0])
            end = tuple(cnt[e, 0])
            far_defect = tuple(cnt[f, 0])
            cv.circle(disp_frame, far_defect, 3, (0, 0, 255), thickness=3)

            # This circle is a part of the gesture number (number of high end convexity defects)
            gesture_number += 1
        b_rect = cv.boundingRect(cnt)
        (x, y, w, h) =b_rect
        cv.rectangle(disp_frame, (x, y), (x + w, y + h), (255, 255, 0))
        # If we've come up this far, then this contour is definitely a hand
        # Write all information about it in a dictionary
        number_hands += 1
        ret_info["result"]["hands"][number_hands - 1] = {
            "hand_number" : number_hands,
            "bounding_rect" : b_rect,
            "number_fingers" : gesture_number + 1
        }

    ret_info["result"]["number_hands"] = number_hands
    if not retInfo:
        return disp_frame
    else:
        return disp_frame, ret_info


kernel = np.ones((5, 5), dtype=np.uint8)
while cam.isOpened():
    # Capture frame and retry if frame dropped
    ret, frame = cam.read()
    if not ret:
        continue

    # Your loop code here
    t1 = cv.getTickCount()
    # Fix things and reduce noise
    frame = cv.flip(frame, 1)
    frame = cv.GaussianBlur(frame, (7, 7), 5)
    # cv.imshow("Live feed", frame)
    view_channels(frame)

    # Stationary objects
    # mask_stationary_obj = stationary_object_mask(frame)
    # cv.imshow("Stationary Objects", mask_stationary_obj)
    pass
    # ~~~~~~~~~~~~~ Stage 1 starts here ~~~~~~~~~~~~~
    # Not skin and skin mask
    mask_not_skin = not_skin_mask(frame)
    cv.imshow("Not Skin mask", mask_not_skin)
    mask_skin = np.zeros_like(mask_not_skin)
    mask_skin[mask_not_skin == 128] = 255
    mask_skin = cv.morphologyEx(mask_skin, cv.MORPH_CLOSE, kernel)
    cv.imshow("Skin mask", mask_skin)
    mask_skin = cv.erode(mask_skin, kernel)
    # ~~~~~~~~~~~~~ Stage 1 ends here ~~~~~~~~~~~~~
    pass
    # ~~~~~~~~~~~~~ Stage 2 starts here ~~~~~~~~~~~~~
    # Perform contour analysis on mask
    disp_frame, hand_information = contour_analysis(retInfo=True)
    # Put up results
    print(len(hand_information["result"]["hands"]), 'hand(s) found : ', end='')
    # ~~~~~~~~~~~~~ Stage 2 ends here ~~~~~~~~~~~~~
    for i in range(hand_information["result"]["number_hands"]):
        # Print hand description
        hand = hand_information["result"]["hands"][i]
        print("{0} giving {1}".format(hand["hand_number"], hand["number_fingers"]), end=', ')
    print('\b\b  ')
    t2 = cv.getTickCount()
    # Measure performance
    time = (t2 - t1)/cv.getTickFrequency()
    cv.putText(disp_frame, "{fps}".format(fps=1/time), (40, 470), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255,255,0))
    cv.imshow("Contours and Hull", disp_frame)

    # disp_frame = frame.copy()
    # face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(frame)
    # for (x, y, w, h) in faces:
    #     cv.rectangle(disp_frame, (x, y), (x + w, y + h), (0, 255, 0))
    # cv.imshow("Faces", disp_frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
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

# Your end code here
cv.destroyAllWindows()
cam.release()
