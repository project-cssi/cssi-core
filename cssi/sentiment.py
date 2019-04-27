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
    NEGATIVE_EMOTIONS = ["angry", "disgust", "scared", "sad"]

    def __init__(self, config, debug):
        super().__init__(debug, config)
        self.face_detector = cv2.CascadeClassifier(self.FACE_DETECTOR_MODEL_PATH)
        self.emotion_detector = load_model(self.EMOTION_DETECTOR_MODEL_PATH, compile=False)
        logger.debug("Sentiment module initialized")

    def generate_final_score(self, all_emotions, expected_emotions):
        """Generators the final sentiment score.

        Different applications will cause the user to portray different emotions.
        ["angry", "disgust", "scared", "sad"] are considered negative emotions by default
        unless specified in the `expected_emotions` array.

        Args:
            all_emotions (list): A list of all the captured emotions
            expected_emotions (list): A list of expected emotions during the session.
        Returns:
            float: The total sentiment score.

        Examples:
            >>> cssi.sentiment.generate_final_score(all_emotions, expected_emotions)
        """
        n_tot = len(all_emotions)  # Total number of emotions captured
        n_neg = 0  # Variable to record the negative emotion count.

        # Checks if the emotion is negative and if it is, and if it is not in
        # the expected emotions list, `n_neg` will be incremented by one.
        for emotion in all_emotions:
            if emotion["sentiment"] in self.NEGATIVE_EMOTIONS:
                if emotion["sentiment"] not in expected_emotions:
                    n_neg += 1

        # Calculating the total sentiment score.
        ts = (n_neg / n_tot) * 100
        return ts

    def generate_sentiment_score(self, frame):
        """Returns the detected emotion."""
        return self.detect_emotions(frame=frame)

    def detect_emotions(self, frame):
        """Detects the sentiment on a face."""
        frame_resized = resize_image(frame, width=300)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            logger.debug("Number of Faces: {0}".format(len(faces)))
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fx, fy, fw, fh) = faces

            # Extract the ROI of the face and resize it to 28x28 pixels
            # to make it compatible with the detector model.
            roi = gray[fy:fy + fh, fx:fx + fw]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = self.emotion_detector.predict(roi)[0]
            label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
            logger.debug("Identified emotion is: {0}".format(label))
            return label
