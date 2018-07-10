import cv2 as cv

cam = cv.VideoCapture(0)

color_mapping = {
    "TrainedClassifiers/Hand.xml": (255, 0, 0),
    "TrainedClassifiers/Hand_0_Gesture.xml": (0, 255, 0)
}
classifiers = {}

for classifier in color_mapping.keys():
    classifiers[classifier] = cv.CascadeClassifier(classifier)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        continue
    frame_disp = frame.copy()
    for classifier in color_mapping.keys():
        clf = classifiers[classifier]
        r_color = color_mapping[classifier]
        d_rects = clf.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in d_rects:
            cv.rectangle(frame_disp, (x, y), (x+w, y+h), r_color)

    cv.imshow("Detection frames", frame_disp)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
cam.release()
