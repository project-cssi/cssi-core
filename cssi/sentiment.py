import cv2
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

from cssi.contributor import CSSIContributor


class Sentiment(CSSIContributor):
    FACE_DETECTOR_MODEL_PATH = "../etc/classifiers/haarcascades/haarcascade_frontalface_default.xml"
    EMOTION_DETECTOR_MODEL_PATH = "../etc/models/_mini_XCEPTION.102-0.66.hdf5"
    POSSIBLE_EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    def __init__(self, expected_emotions, debug=False):
        self.expected_emotions = expected_emotions
        self.debug = debug
        self._init_classifiers()

    def score(self, emotion):
        print(emotion)

    def _init_classifiers(self):
        self.face_detector = cv2.CascadeClassifier(self.FACE_DETECTOR_MODEL_PATH)
        self.emotion_detector = load_model(self.EMOTION_DETECTOR_MODEL_PATH, compile=False)

    def detect_emotions(self, frame):
        frame_resized = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # Extract the ROI of the face from the grayscale image,
            # resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the model
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = self.emotion_detector.predict(roi)[0]
            confidence = np.max(predictions)
            label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
            return label