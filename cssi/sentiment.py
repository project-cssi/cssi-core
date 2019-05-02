import os
import logging
import cv2
from pathlib import Path
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

from cssi.contributor_base import CSSIContributor
from cssi.utils.image_processing import resize_image, prep_image

logger = logging.getLogger(__name__)


class Sentiment(CSSIContributor):
    FACE_DETECTOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            Path("data/models/res10_300x300_ssd_iter_140000.caffemodel"))
    CAFFE_PROTO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         Path("data/helper/deploy.prototxt.txt"))
    EMOTION_DETECTOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               Path("data/models/_mini_XCEPTION.102-0.66.hdf5"))

    POSSIBLE_EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    NEGATIVE_EMOTIONS = ["angry", "disgust", "scared", "sad"]

    def __init__(self, config, debug):
        super().__init__(debug, config)
        self.face_detector = cv2.dnn.readNetFromCaffe(self.CAFFE_PROTO_FILE_PATH, self.FACE_DETECTOR_MODEL_PATH)
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
        # prepare the frame for processing
        frame = prep_image(frame)

        frame = resize_image(frame, width=400)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (64, 64))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            predictions = self.emotion_detector.predict(face)[0]
            label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
            logger.debug("Identified emotion is: {0}".format(label))

            # draw the bounding box of the face along with the associated
            # probability
            text = "{0}: {1:.2f}%".format(label, confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow('Sentiment', frame)
            cv2.waitKey(0)
            return label
