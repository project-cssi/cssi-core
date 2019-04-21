from cssi.questionnaire import SSQ
from cssi.latency import Latency
from cssi.sentiment import Sentiment
from cssi.core import CSSI
import cv2
import time

"""Questionnaire Module Testing"""
PRE = {
    "blurredVision": 1,
    "burping": 2,
    "difficultyConcentrating": 3,
    "difficultyFocusing": 1,
    "dizzyEyesClosed": 1,
    "dizzyEyesOpen": 0,
    "eyestrain": 0,
    "fatigue": 1,
    "fullnessOfHead": 0,
    "generalDiscomfort": 0,
    "headache": 1,
    "increasedSalivation": 2,
    "nausea": 2,
    "stomachAwareness": 0,
    "sweating": 1,
    "vertigo": 0
}

POST = {
    "blurredVision": 1,
    "burping": 2,
    "difficultyConcentrating": 3,
    "difficultyFocusing": 2,
    "dizzyEyesClosed": 1,
    "dizzyEyesOpen": 0,
    "eyestrain": 0,
    "fatigue": 1,
    "fullnessOfHead": 3,
    "generalDiscomfort": 0,
    "headache": 1,
    "increasedSalivation": 2,
    "nausea": 2,
    "stomachAwareness": 3,
    "sweating": 1,
    "vertigo": 0
}


cssi = CSSI("config.cssi")
ssq = SSQ(cssi.config, False)
scores = ssq.generate_score(pre=PRE, post=POST)
# print("POST SYMPTOM SCORES : N - {0}, O - {1}, D - {2}".format(N, D, O))
print(scores)

"""Latency Module Testing"""
latency = Latency(debug=False, config=None,)
sentiment = Sentiment(debug=False, config=None, expected_emotions=["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"])

first_frame = cv2.imread("../etc/test_assets/5.JPG")
second_frame = cv2.imread("../etc/test_assets/6.JPG")

first_frame, second_frame, pitch, yaw, roll = latency.calculate_camera_pose(first_frame=first_frame, second_frame=second_frame)
print("Camera Pitch: {0}, Yaw: {1}, Roll: {2}".format(pitch, yaw, roll))

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)  # to give time to the camera to warm up

# Create the main window and move it
cv2.namedWindow("Video")
cv2.moveWindow("Video", 20, 20)

while True:
    _, frame = video_capture.read()
    frame_mod, pitch, yaw, roll = latency.calculate_head_pose(frame)
    emotion = sentiment.detect_emotions(frame=frame)
    sentiment.generate_score(emotion=emotion)

    print("Face Pitch: {0}, Yaw: {1}, Roll: {2}".format(pitch, yaw, roll))

    # Showing the frame and waiting
    # for the exit command
    cv2.imshow("Video", frame_mod)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Release the camera
video_capture.release()
print("Bye...")


