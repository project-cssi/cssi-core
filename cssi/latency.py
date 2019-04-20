#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This can be used to measure head pose estimation using
68 facial landmark detection along with pretrained hog and linear svm
face detection in dlib.

"""

import cv2
import dlib
import numpy as np
from cssi.contributor import CSSIContributor
from cssi.utils import calculate_euler_angles


class Latency(CSSIContributor):

    def __init__(self, timeout, debug=False):
        self.timeout = timeout
        self.debug = debug

    def score(self):
        print("score")

    def calculate_head_pose(self, frame):
        hp = HeadPoseCalculator(frame, self.debug)
        return hp.calculate_head_pose()


class HeadPoseCalculator(object):

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor("../etc/shape_predictor_68_face_landmarks.dat")

    def __init__(self, frame, debug=False):
        self.frame = frame
        self.debug = debug

    def calculate_head_pose(self):
        # convert the frame to gray-scale
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Looking for faces with dlib get_frontal_face detector in the gray scale frame
        faces = self.face_detector(gray, 0)

        # ONLY IN DEBUG MODE: Draw number of faces on frame
        self._draw_face_num(faces=faces)

        pitch, yaw, roll = 0.0, 0.0, 0.0

        # loop over the face detections
        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

            # ONLY IN DEBUG MODE: Draw a green rectangle (and text) around the face.
            self._draw_face_rect(top=top, left=left, right=right, bottom=bottom)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = self.landmark_detector(gray, face)
            landmark_coords = np.zeros((shape.num_parts, 2), dtype="int")

            # 2D model points
            image_points = np.float32([
                (shape.part(30).x, shape.part(30).y),  # nose
                (shape.part(8).x, shape.part(8).y),  # Chin
                (shape.part(36).x, shape.part(36).y),  # Left eye left corner
                (shape.part(45).x, shape.part(45).y),  # Right eye right corner
                (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
                (shape.part(54).x, shape.part(54).y),  # Right mouth corner
                (shape.part(27).x, shape.part(27).y)
            ])

            # 3D model points
            model_points = np.float32([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (225.0, 170.0, -135.0),  # Left eye left corner
                (-225.0, 170.0, -135.0),  # Right eye right corner
                (150.0, -150.0, -125.0),  # Left Mouth corner
                (-150.0, -150.0, -125.0),  # Right mouth corner
                (0.0, 140.0, 0.0)
            ])

            # image properties. channels is not needed so _ is to drop the value
            height, width, _ = self.frame.shape

            # Camera internals double
            focal_length = width
            center = np.float32([width / 2, height / 2])
            camera_matrix = np.float32([[focal_length, 0.0, center[0]],
                                        [0.0, focal_length, center[1]],
                                        [0.0, 0.0, 1.0]])
            dist_coeffs = np.zeros((4, 1), dtype="float32")  # Assuming no lens distortion

            retval, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            nose_end_point3D = np.float32([[50, 0, 0],
                                           [0, 50, 0],
                                           [0, 0, 50]])

            nose_end_point2D, jacobian = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)

            camera_rot_matrix, _ = cv2.Rodrigues(rvec)

            pitch, yaw, roll = calculate_euler_angles(R=camera_rot_matrix)

            # ONLY IN DEBUG MODE: Draw points used head pose estimation
            self._draw_face_points(points=image_points)

            # ONLY IN DEBUG MODE: Draw landmarks used head pose estimation
            self._draw_face_landmarks(coords=landmark_coords, shape=shape)

            # ONLY IN DEBUG MODE: Write the euler angles on the frame
            self._draw_angles(pitch=pitch, yaw=yaw, roll=roll, left=left, bottom=bottom)

        return [self.frame, pitch, yaw, roll]

    def _draw_face_num(self, faces):
        """Draw number of faces on frame.

        Check to see if a face was detected,
        and if so, draw the total number of faces on the frame

        """
        if self.debug:
            if len(faces) > 0:
                text = "{0} face(s) found".format(len(faces))
                cv2.putText(self.frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

    def _draw_face_rect(self, top, left, right, bottom):
        """Draw a green rectangle (and text) around the face."""
        if self.debug:
            label_x = left
            label_y = top - 3
            if label_y < 0:
                label_y = 0
            cv2.putText(self.frame, "FACE", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(self.frame, (left, top), (right, bottom),
                          (0, 255, 0), 1)

    def _draw_face_points(self, points):
        """Draw used points for head pose estimation"""
        if self.debug:
            for point in points:
                cv2.circle(self.frame, (point[0], point[1]), 3, (255, 0, 255), -1)

    def _draw_face_landmarks(self, coords, shape):
        """Draw the landmarks used for head pose on the frame."""
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        for (i, (x, y)) in enumerate(coords):
            cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(self.frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def _draw_angles(self, pitch, yaw, roll, left, bottom):
        """Write the euler angles on the frame"""
        cv2.putText(self.frame, "Pitch: {}".format(pitch), (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(self.frame, "Yaw: {}".format(yaw), (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(self.frame, "Roll: {}".format(roll), (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)
