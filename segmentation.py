from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


class Segmentator(object):
    """
    Класс предназначен на разбивание изображение на отдельные участки: нос, рот, правый глаз, левый глаз 
    """
    def __init__(self,model = 'model/shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model)

    def process(self, img):
        """
        На вход получает изображение, возвращает словарь нос, губы, глаза, брови, линия подбородка и все лицо,
        каждое поле словаря состоит из y,x,h,w  - координат точки, ширины и длины
        """

        out_dict = {}

        image = cv2.imread(img)
        # Нужен ли он нам?
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                clone = image.copy()
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                # roi = image[y:y + h, x:x + w]
                # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                # print(name)
                out_dict[name] = [x,y,w,h]
                #cv2.imshow("ROI", roi)
                #cv2.waitKey(0)
            return out_dict


segm = Segmentator()
print(segm.process('test.jpg'))