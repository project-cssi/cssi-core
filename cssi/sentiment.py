import os
import logging
import cv2
from pathlib import Path
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

from cssi.contributor import CSSIContributor
from cssi.utils.image_processing import resize_image

logger = logging.getLogger(__name__)


class Sentiment(CSSIContributor):
    FACE_DETECTOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("data/classifiers/haarcascades/haarcascade_frontalface_default.xml"))
    EMOTION_DETECTOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("data/models/_mini_XCEPTION.102-0.66.hdf5"))
    POSSIBLE_EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    def __init__(self, config, debug):
        super().__init__(debug, config)
        self.face_detector = cv2.CascadeClassifier(self.FACE_DETECTOR_MODEL_PATH)
        self.emotion_detector = load_model(self.EMOTION_DETECTOR_MODEL_PATH, compile=False)
        logger.debug("Sentiment module initialized")

    def generate_final_score(self, emotions):
        logger.debug("All Emotions: {}".format(emotions))

    def generate_unit_score(self, emotion):
        logger.debug("Sentiment Score: {}".format(emotion))

    def detect_emotions(self, frame):
        frame_resized = resize_image(frame, width=300)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            logger.debug("Number of Faces: {0}".format(len(faces)))
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # Extract the ROI of the face and resize it to 28x28 pixels
            # to make it compatible with the detector model.
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = self.emotion_detector.predict(roi)[0]
            # confidence = np.max(predictions)
            logger.debug("Possible emotions: {0}".format(self.POSSIBLE_EMOTIONS))
            label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
            logger.debug("Identified emotion is: {0}".format(label))
            print(label)
            return label
