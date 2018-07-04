import cv2 as cv
import numpy as np
import pickle
import _thread
from MachineLearningSolutions.NeuralNetworks import ANN_Trial3 as NN

fname = "../MachineLearningSolutions/NeuralNetworks/Results/NN_TEST_5_FINGERS_4-6-2018_"
with open(fname, 'rb') as fs:
    (layers, activation_functions), params = pickle.load(fs)
print(layers, activation_functions)
activation_functions = NN.get_activations_list(activation_functions)
print(activation_functions)

cam = cv.VideoCapture(0)


def generate_mask(frame):
    """
    Generates a mask of the region that may be skin
    :param frame:
    :return:
    """
    greater_red_mask = np.ones_like(frame[:, :, 2], dtype=np.uint8) * 255
    greater_red_mask[frame[:, :, 0] + 10 >= frame[:, :, 2]] = 0
    greater_red_mask[frame[:, :, 1] + 10 >= frame[:, :, 2]] = 0
    img_hsv = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)
    _, hue_mask = cv.threshold(img_hsv[:, :, 0], 11, 255, cv.THRESH_BINARY)
    ret_mask = greater_red_mask
    ret_mask[hue_mask > 127] = 0
    return ret_mask


def sliding_window(mask, sw=1, sh=1):
    # Take a window of size 235 rows by 190 columns and slide it over the mask (of size 480 rows by 640 columns)
    def get_frame_prediction(frame):
        """
        Get prediction on the single sliding window
        :param frame: 235 by 190 frame
        :return:
            The prediction of the neural network
        """
        image = frame.copy()
        image = cv.resize(image, None, fx=1/5, fy=1/5)
        _, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        x_input = np.reshape(image, (-1, 1))
        prediction, _ = NN.forward_propagate_deep(params, activation_functions, x_input)
        prediction = np.floor(prediction * 255)
        return prediction
    # Get the shape of the mask
    (height, width) = mask.shape
    res_widht = int((width - 190)/sw + 1)
    res_height = int((height - 235)/sh + 1)
    ret_heat_map = np.zeros((res_height, res_widht), dtype=np.uint8)
    r_f = 0
    c_f = 0
    for r_no in range(0, height, sh):
        r_end = r_no + 235
        if r_end > height:
            break
        for c_no in range(0, width, sw):
            c_end = c_no + 190
            if c_end > width:
                break
            else:   # Perform forward prop
                in_frame = mask[r_no:r_end, c_no:c_end]
                sub_val = get_frame_prediction(in_frame)
                ret_heat_map[r_f, c_f] = sub_val
                c_f += 1
        r_f += 1
        c_f = 0 # Reset the column count
    return ret_heat_map

stride_h = 10
stride_w = 15


def get_point_on_image(mx, my):
    # Row, column
    return int((mx * stride_h + 117 + 75) * 4/3), int((my * stride_w + 94) * 4/3)


def save_rect_as_false_example():
    # The image shown is not a hand, save the mask in "Data" directory with name "False"
    with open('Data/Index.txt', 'r') as f:
        ind = int(f.read())
    im_save = mask[pimg[0] - 117:pimg[0] + 117, pimg[1] - 94:pimg[1] + 94]
    im_save = cv.resize(im_save, (190, 235))
    cv.imwrite("Data/False{no}.jpg".format(no=ind), im_save)
    print("File \"Data/False{no}.jpg\"saved, shape is {fsh}".format(fsh=im_save.shape,
                                                                    no=ind))
    with open('Data/Index.txt', 'w') as f:
        f.write(str(ind + 1))


while cam.isOpened():
    c1 = cv.getTickCount()
    ret, img = cam.read()
    img = cv.GaussianBlur(img, (5, 5), 3)
    img_scaled_down = cv.resize(img, None, fx=3 / 4, fy=3 / 4)
    mask = generate_mask(img_scaled_down)
    # Remove all contours that are not big enough to be a hand
    mask, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    draw_mask = mask
    draw_mask = cv.cvtColor(draw_mask, cv.COLOR_GRAY2BGR)
    area_threshold = 1900
    eligible_cnts = []
    for cnt in contours:
        if cv.contourArea(cnt) < area_threshold:
            cv.drawContours(draw_mask, [cnt], 0, (0, 0, 0), thickness=-1)
        else:
            eligible_cnts.append([cnt])
    mask = draw_mask[:, :, 0]
    heat_map = sliding_window(mask, sw=stride_w, sh=stride_h)
    max_index = np.argmax(heat_map)
    (h, w) = heat_map.shape
    (max_prob_x, max_prob_y) = (max_index // w, max_index % w)
    pimg = get_point_on_image(max_prob_x, max_prob_y)
    cv.rectangle(img, (pimg[1] - 94, pimg[0] - 117), (pimg[1] + 94, pimg[0] + 117), (0, 255, 0))
    cv.imshow("Original Image", img)
    cv.imshow("Mask", mask)
    cv.imshow("Heat map", heat_map)
    cv.imshow("Prediction Rectangle", img[pimg[0] - 117:pimg[0] + 117, pimg[1] - 94:pimg[1] + 94, :])
    c2 = cv.getTickCount()
    t = (c2 - c1) / cv.getTickFrequency()
    f = 1 / t
    key = cv.waitKey(1) & 0xff
    if key == ord('q') or key == 27:
        break
    elif key == ord('p'):
        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord('p'):
                break
            if key == ord('n'):
                save_rect_as_false_example()
    elif key == ord('n'):
        save_rect_as_false_example()

cv.destroyAllWindows()
cam.release()
