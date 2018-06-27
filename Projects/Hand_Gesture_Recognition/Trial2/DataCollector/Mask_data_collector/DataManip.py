"""
Data collector/scrapper tool

Made by : Avneesh Mishra
"""
import os
import cv2 as cv
import numpy as np

data_folder_name = "Data"

# Place where all the flipped images are placed
data_dest_folder_name = "{r}_augmented".format(r=data_folder_name)
try:
    os.mkdir(data_dest_folder_name)
except FileExistsError:
    print("Warning : {dest} -> Directory already exists".format(dest=data_dest_folder_name))
    print("You will loose all the data in the directory if you choose to proceed")
    while True:
        user_choice = input('Proceed ? [y/n] : ')
        if user_choice == 'n':
            print("Exit code received")
            exit(0)
        elif user_choice == 'y':
            print("Removing all contents inside file \"{fname}\"".format(fname=data_dest_folder_name))
            os.system("rm -r {dest}".format(dest=data_dest_folder_name))
            os.mkdir(data_dest_folder_name)
            break
        else:
            print("Invalid input")

declared_data_variables = False
label = 0
num_lb = len(os.listdir("{r}/".format(r=data_folder_name)))
print("{num_labels} labels found".format(num_labels= num_lb))
# Scrap all data
# Iterate over every folder in data folder
for label_directory in sorted(os.listdir("{r}/".format(r=data_folder_name))):
    print("INFO : {label_dir} labelled {l_no}, row number {r_no}".format(label_dir=label_directory, l_no=label, r_no=label+1))
    num = 0
    # Iterate over every image file
    for file_name in os.listdir("{r}/{lb}/".format(r=data_folder_name, lb=label_directory)):
        img = cv.imread("{r}/{lb}/{f}".format(r=data_folder_name,
                                             lb=label_directory, f=file_name))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_flip = cv.flip(img, 1)
        # Show image
        cv.imshow("Original Image", img)
        cv.imshow("Flipped Image", img_flip)
        cv.moveWindow("Original Image", 50, 60)
        cv.moveWindow("Flipped Image", 500, 60)
        # Make directory for image
        if num == 0:
            dname = '{r}/{lb}'.format(r=data_dest_folder_name, lb=label_directory)
            os.mkdir(dname)
            print("DATA DEBUG : Making directory \"{dirname}\"".format(dirname=dname))

        # Save main image
        cv.imwrite('{r}/{lb}/M_{n}.jpg'.format(r=data_dest_folder_name, lb=label_directory, n=2 * num), img)
        # Save flipped image
        cv.imwrite('{r}/{lb}/M_{n}.jpg'.format(r=data_dest_folder_name, lb=label_directory, n=2 * num + 1), img_flip)
        # Saving data as variables
        img = cv.resize(img, None, fx=1/5, fy=1/5)
        img_flip = cv.resize(img_flip, None, fx=1/5, fy=1/5)
        if not declared_data_variables:
            print("DATA DEBUG : Saving images of shape {im_size} -> {num_cols} columns".format(im_size=img.shape,
                                                                            num_cols=img.reshape((-1,1)).shape[0]))
            # Image (input data)
            X = np.array(np.reshape(img, (-1,1)))
            X = np.array(np.column_stack((X, np.array(np.reshape(img_flip, (-1,1))))))
            # Single row (output data in form of numbers)
            Y = np.array([label])
            Y = np.column_stack((Y,Y))
            # Multiple row (one hot encoding)
            Y_one_hot = np.zeros((num_lb, 1))
            Y_one_hot[label] = 1
            Y_one_hot = np.column_stack((Y_one_hot, Y_one_hot))
            declared_data_variables = True
        else:
            # Image (input data)
            stacked_images = np.column_stack((img.reshape((-1,1)), img_flip.reshape((-1,1))))
            X = np.column_stack((X, stacked_images))
            # Single row (output data in form of numbers)
            Y_img = np.array([label])
            Y_col = np.column_stack((Y_img, Y_img))
            Y = np.column_stack((Y, Y_col))
            # Multiple row (one hot encoding)
            Y_one_hot_buffer = np.zeros((num_lb, 1))
            Y_one_hot_buffer[label] = 1
            Y_one_hot_buffer = np.column_stack((Y_one_hot_buffer, Y_one_hot_buffer))
            Y_one_hot = np.column_stack((Y_one_hot, Y_one_hot_buffer))
        num += 1
        # Ending code
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):  # Skip this folder
            break
        elif key == ord('i'):  # Information about a file
            print("IMAGE INFO : File name : \"{f}\", shape : {s}, flat shape = {fs}".format(
                f=file_name, s=img.shape, fs=np.reshape(img, [-1, 1]).shape))
        elif key == 27:  # Escape key pressed
            exit(0)
        elif key == ord('p'):  # Pause
            input("Press enter to continue...")
    label += 1
    print("DATA DEBUG : {nf} files saved to directory \"{dirname}\"".format(nf=2*num, dirname=dname))

print("INFO : Input shape : {in_shape}, Output shape (Y) : {out_shape}, One hot encoding shape : {one_hot_shape}".format(
    in_shape= X.shape, out_shape= Y.shape, one_hot_shape=Y_one_hot.shape))
cv.destroyAllWindows()
data_dest_folder_name = "{r}_numpy_files".format(r=data_folder_name)
try:
    os.mkdir(data_dest_folder_name)
except FileExistsError:
    print("Warning : {dest} -> Directory already exists".format(dest=data_dest_folder_name))
    print("You will loose all the data in the directory if you choose to proceed")
    while True:
        user_choice = input('Proceed ? [y/n] : ')
        if user_choice == 'n':
            print("Exit code received")
            exit(0)
        elif user_choice == 'y':
            print("Removing all contents inside file \"{fname}\"".format(fname=data_dest_folder_name))
            os.system("rm -r {dest}".format(dest=data_dest_folder_name))
            os.mkdir(data_dest_folder_name)
            break
        else:
            print("Invalid input")

print("DATA DEBUG : Saving this data to \"{dest}\"".format(dest=data_dest_folder_name))
np.save('{dst}/X.npy'.format(dst=data_dest_folder_name), X)
np.save('{dst}/Y.npy'.format(dst=data_dest_folder_name), Y)
np.save('{dst}/Y_one_hot_encoded.npy'.format(dst=data_dest_folder_name), Y_one_hot)
print("Data saved")
