import numpy as np
import cv2

camera = cv2.VideoCapture(0)

with open('Snapshots/index.txt', 'r') as fr:
    index = int(fr.readline())


while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    viewFrame = np.copy(frame)
    cv2.putText(viewFrame, 'Press "c" to capture', (75,475), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), thickness=1)
    cv2.imshow('Frames', viewFrame)

    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q') or keyPressed == 27:
        camera.release()
        cv2.destroyAllWindows()
        break
    elif keyPressed == ord('c'):
        fileName = './Snapshots/image{0}.jpg'.format(index)
        print('Snapshot saved as "',fileName,'"')
        cv2.imwrite(filename= fileName, img= frame)
        index += 1


with open('Snapshots/index.txt', 'w') as f:
    f.write(str(index))