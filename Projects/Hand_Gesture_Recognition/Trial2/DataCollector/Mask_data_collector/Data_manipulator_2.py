import numpy as np
import cv2 as cv
import os

prime_file = "4_fingers"
root_dir = "Data_augmented"
num_labels = len(os.listdir("{r}/".format(r=root_dir)))
# Must be a number between 0 and 100
num_files = 0
m_num_f = 0
for filename in os.listdir("{r}/".format(r=root_dir)):
    n_f = len(os.listdir("{r}/{f}/".format(r=root_dir, f=filename)))
    if filename == prime_file:
        m_num_f = n_f
    print("{f_n} files in {f}".format(f_n = n_f, f=filename))
    num_files += n_f
print("{number_files} total files".format(number_files=num_files))

f_req = 0.5     # Fraction required

print("{freq} files required".format(freq = m_num_f * 1/f_req))
num_files_each_label = m_num_f * (1/f_req - 1) / (num_labels - 1)
num_files_each_label = int(np.ceil(num_files_each_label))
print("{ireq} images from each label".format(ireq=num_files_each_label))
# Read m_num_f from prime_file and read num_files_each_label from other folders
try:
    os.mkdir("Data_Distribution_Generated")
except FileExistsError:
    ch = input("Delete existing file ? [Y/N] : ")
    if ch == 'y' or ch == 'Y':
        os.system("rm -r Data_Distribution_Generated")
        os.mkdir("Data_Distribution_Generated")
    else:
        exit(0)
os.mkdir("Data_Distribution_Generated/True")
os.mkdir("Data_Distribution_Generated/False")
os.mkdir("Data_Distribution_Generated/NP_Files")

data_vars_initialized = False
for filename in os.listdir("{r}/".format(r=root_dir)):
    if filename == prime_file:
        flag = "True"
    else:
        flag = "False"
    n = 0
    for file in os.listdir("{r}/{f}/".format(r=root_dir, f=filename)):
        # print("{r}/{f}/{fl}".format(r=root_dir, f=filename, fl=file))
        img = cv.imread("{r}/{f}/{fl}".format(r=root_dir, f=filename, fl=file))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("Image", img)
        cv.imwrite("Data_Distribution_Generated/{fl}/{f}_{num}.jpg".format(
            fl=flag, f=filename, num = n), img)
        n += 1
        # Numpy save
        img = cv.resize(img, None, fx=1/5, fy=1/5)
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        if not data_vars_initialized:
            X = np.ndarray.flatten(img).reshape((-1,1))
            if flag == "True":
                Y = np.array([[1]])
            else:
                Y = np.array([[0]])
            data_vars_initialized = True
        else:
            s_X = np.ndarray.flatten(img).reshape((-1, 1))
            X = np.column_stack((X, s_X))
            if flag == "True":
                s_Y = np.array([[1]])
            else:
                s_Y = np.array([[0]])
            Y = np.column_stack((Y, s_Y))
        if flag == "False" and n == num_files_each_label:
            break
        elif flag == "True" and n == m_num_f:
            break
        key = cv.waitKey(1) & 0xff
        if key == 27 or key == ord('q'):
            exit(0)
print("INPUT SHAPE : {inshape}".format(inshape=X.shape))
print("OUTPTU SHAPE : {outshape}".format(outshape=Y.shape))
np.save("Data_Distribution_Generated/NP_Files/X.npy", X)
np.save("Data_Distribution_Generated/NP_Files/Y.npy", Y)
