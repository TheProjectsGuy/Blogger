import os

data_dir = "Data/"
print(os.listdir(data_dir))

# Folder Not_fingers
num_Es_nf = len([filename for filename in os.listdir("Data/Not_fingers") if filename.startswith('E')])
print(num_Es_nf)

img_images = [filename for filename in os.listdir("Data/") if filename.startswith('IMG')]
print(img_images)
for filename in img_images:
    os.rename('Data/{fn}'.format(fn=filename), 'Data/Not_fingers/E_{idno}.jpg'.format(idno= num_Es_nf))
    num_Es_nf += 1

num_HANDs_ges = len([filename for filename in os.listdir("Data/Gestures") if filename.startswith('E')])
print(num_HANDs_ges)

hand_images = [filename for filename in os.listdir("Data/") if filename.startswith('HAND')]
print(hand_images)
for filename in hand_images:
    os.rename('Data/{fn}'.format(fn=filename), 'Data/Gestures/E_{idno}.jpg'.format(idno= num_HANDs_ges))
    num_HANDs_ges += 1
