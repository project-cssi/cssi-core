#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This can be used to measure head pose estimation using
68 facial landmark detection along with pretrained hog and linear svm
face detection in dlib.

"""
import os
import logging
import cv2
import dlib
import numpy as np
from pathlib import Path

from cssi.contributor import CSSIContributor
from cssi.utils.physics import calculate_euler_angles, calculate_angle_diff
from cssi.utils.image_processing import split_image_in_half, resize_image

logger = logging.getLogger(__name__)


class Latency(CSSIContributor):
    FACE_DETECTOR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            Path("data/models/res10_300x300_ssd_iter_140000.caffemodel"))
    CAFFE_PROTO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         Path("data/helper/deploy.prototxt.txt"))

    ANGLE_MEASUREMENT_ERR = 3
    DEFAULT_LATENCY_BOUNDARY = 20

    def __init__(self, config, debug, shape_predictor):
        """Creates an instance of the latency class.

        This class is automatically instantiated by the `CSSI` class and the class overrides
        the parent class i.e `CSSIContributor` constructor and initializes `face_detector`,
        `landmark_detector` and `feature_detector`.
        Args:
            config (object): An object of the Config class.
            debug (bool): Boolean specifying if debug is enabled or not.
            shape_predictor (str): Path to the landmark detector file.
        """
        super().__init__(config=config, debug=debug)
        self.face_detector = cv2.dnn.readNetFromCaffe(self.CAFFE_PROTO_FILE_PATH, self.FACE_DETECTOR_MODEL_PATH)
        self.landmark_detector = dlib.shape_predictor(shape_predictor)
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        logger.debug("Latency module initialized")

    def generate_final_score(self, scores):
        """Generators the final latency score.

        `sum_ln` is used to persist the sum of the individual latency scores.
        Then the sum is divided by n`, which is the number of latency tests carried out.
        The result is then multiplied by 100 to generate `tl` (Total Latency Score).

        Args:
            scores (list): A list of python dictionaries containing all the individual
                latency scores. ex: [{"score": 0,"timestamp": "2019-04-24 18:29:25"}]

        Returns:
            float: The total latency score.

        Examples:
            >>> cssi.latency.generate_final_score(scores)
        """
        n = len(scores)  # Total number of emotions captured
        sum_ls = 0.0  # Variable to store thr sum of the individual latency scores

        # Calculates the sum of latency scores.
        for score in scores:
            sum_ls += score['score']

        # Calculating the total latency score i.e `tl`
        tl = (sum_ls / n) * 100
        return tl

    def generate_rotation_latency_score(self, head_angles, camera_angles):
        """Evaluates the latency score for a corresponding head and scene rotation pair.

        Args:
            head_angles (list): Pair of head rotation angles in a list. i.e pitch, yaw and
                roll of previous and current rotation.
            camera_angles(list): Pair of scene rotation angles in a list. i.e pitch, yaw and
                roll of previous and current rotation.

        Returns:
            int: If there is a discrepancy in the rotations 1 will be returned, else 0.

        Examples:
            >>> cssi.latency.generate_rotation_latency_score(head_angles, camera_angles)
        """
        # Calculates the difference of the angle pairs.
        hp_diff, hy_diff, hr_diff = self._calculate_angle_pair_diff(head_angles)
        cp_diff, cy_diff, cr_diff = self._calculate_angle_pair_diff(camera_angles)

        # Checks if the difference between the angles is greater than the measurement error.
        # If yes for either pitch, yaw or roll difference, 1 will be returned, else 0.
        if abs(calculate_angle_diff(hp_diff, cp_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        elif abs(calculate_angle_diff(hy_diff, cy_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        elif abs(calculate_angle_diff(hr_diff, cr_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        return 0

    def generate_pst_latency_score(self, head_stream, camera_stream):
        """Evaluates the latency score based on the `Pixel Switching Times` (pst).

        This function first check if there is a head movement in the passed in head frame stream
        and if there is, it calculates the `Pixel Switching Times` (pst) of frames in the camera
        frame stream.

        Args:
            head_stream (list): List of head frames.
            camera_stream(list): List of camera frames(scene frames).

        Returns:
            int: If the pst is more than the motion-to-photon latency boundary which is specified
                in the configuration (default 20ms), a score of 1 will be returned. If there is no head
                movement or if the pst is less than the boundary, 0 will be returned.

        Examples:
            >>> cssi.latency.generate_pst_latency_score(head_stream, camera_stream)
        """
        # Check if there is a head movement
        movement = self.check_for_head_movement(stream=head_stream)

        # If there is no movement, returns 0.
        if not movement:
            return 0

        # Calculates the `Pixel Switching Times` (pst)
        pst = self.calculate_pst(stream=camera_stream)

        # If the pst is greater than the latency boundary, 1 will be returned. Else 0. And if
        # the pst is `None`, 0 will be returned.
        if pst is not None:
            if pst > self.config.latency_boundary or self.DEFAULT_LATENCY_BOUNDARY:
                return 1
            else:
                return 0
        else:
            return 0

    def check_for_head_movement(self, stream):
        """Checks whether if there is a head movement in a stream of head frames."""
        phf_pitch, phf_yaw, phf_roll = 0.0, 0.0, 0.0
        for idx, frame in enumerate(stream):
            _, chf_pitch, chf_yaw, chf_roll = self.calculate_head_pose(frame=frame)
            if idx != 0:
                if abs(calculate_angle_diff(angle_1=phf_pitch, angle_2=chf_pitch)) > 0:
                    return True
                elif abs(calculate_angle_diff(angle_1=phf_yaw, angle_2=chf_yaw)) > 0:
                    return True
                elif abs(calculate_angle_diff(angle_1=phf_roll, angle_2=chf_roll)) > 0:
                    return True
            phf_pitch, phf_yaw, phf_roll = chf_pitch, chf_yaw, chf_roll
        return False

    @staticmethod
    def calculate_pst(stream, fps):
        """Calculates the `Pixel Switching Times` (pst) of a camera frame stream."""
        prev_frame = None
        processed_count = 0
        equal_count = 0
        for idx, frame in enumerate(stream):
            processed_count += 1
            if idx != 0:
                diff = cv2.subtract(prev_frame, frame)
                B, G, R = cv2.split(diff)
                # If all the pixels (Red, Green &  Blue) are equal then the two images are similar.
                # If not then the images are different and the pst is calculated.
                if cv2.countNonZero(B) == 0 and cv2.countNonZero(G) == 0 and cv2.countNonZero(R) == 0:
                    equal_count += 1
                else:
                    return (processed_count / fps) * 1000
            prev_frame = frame
        # If the stream did not have any different frames, `None` will be returned.
        return None

    def calculate_head_pose(self, frame):
        """Returns the head rotation angles.

        Args:
            frame (list): An image frame.(numpy array)

        Returns:
            tuple: [`frame`, `pitch`, `yaw`, `roll`] will be returned as a tuple.

        Examples:
            >>> cssi.latency.calculate_head_pose(frame)
        """
        hp = HeadPoseCalculator(debug=self.debug, frame=frame, landmark_detector=self.landmark_detector, face_detector=self.face_detector)
        return hp.calculate_head_pose()

    def calculate_camera_pose(self, first_frame, second_frame, crop=True, crop_direction='horizontal'):
        """Returns the head rotation angles.

        Args:
            first_frame (list): First scene frame.(numpy array)
            second_frame (list): Second scene frame.(numpy array)
            crop (bool): Specifies if the frame should be cropped or not.
            crop_direction (str): Specifies if the frame should be split horizontally or vertically.

        Returns:
            tuple: [`first_frame`, `second_frame`, `pitch`, `yaw`, `roll`] will be returned as a tuple.

        Examples:
            >>> cssi.latency.calculate_camera_pose(first_frame,second_frame, True, 'horizontal')
        """
        # if crop is true, split the image in two and take the
        # first part and sent it to pose calculator.
        if crop:
            first_frame, _ = split_image_in_half(image=first_frame, direction=crop_direction)
            second_frame, _ = split_image_in_half(image=second_frame, direction=crop_direction)
        cp = CameraPoseCalculator(debug=self.debug, first_frame=first_frame, second_frame=second_frame, feature_detector=self.feature_detector)
        return cp.calculate_camera_pose()

    @staticmethod
    def _calculate_angle_pair_diff(angles):
        """Calculates the difference between the pitch, yaw and roll of a pair of rotation angle sets."""
        f1_rot = angles[0]
        f2_rot = angles[1]

        pitch_diff = calculate_angle_diff(f2_rot[0], f1_rot[0])
        pitch_yaw = calculate_angle_diff(f2_rot[1], f1_rot[1])
        pitch_roll = calculate_angle_diff(f2_rot[2], f1_rot[2])

        return pitch_diff, pitch_yaw, pitch_roll


class HeadPoseCalculator(object):

    def __init__(self, debug, frame, landmark_detector, face_detector):
        self.debug = debug
        self.frame = frame
        self.landmark_detector = landmark_detector
        self.face_detector = face_detector

    def calculate_head_pose(self):
        """Detects the sentiment on a face."""
        self.frame = resize_image(self.frame, width=400)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        (h, w) = self.frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # ONLY IN DEBUG MODE: Draw number of faces on frame
        # self._draw_face_num(faces=faces)

        pitch, yaw, roll = 0.0, 0.0, 0.0

        # loop over the face detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # ONLY IN DEBUG MODE: Draw a green rectangle (and text) around the face.
            self._draw_face_rect(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)

            face = dlib.rectangle(left=start_x, top=start_y, right=end_x, bottom=end_y)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = self.landmark_detector(gray, face)

            # 2D model points
            image_points = np.float32([
                (shape.part(7).x, shape.part(7).y),  # 33 - nose bottom
                (shape.part(34).x, shape.part(34).y),  # 8 - chin
                (shape.part(11).x, shape.part(11).y),  # 54 - lip right
                (shape.part(18).x, shape.part(18).y),  # 48 - lip left
                (shape.part(4).x, shape.part(4).y),  # 3 - ear right
                (shape.part(3).x, shape.part(3).y),  # 13 - ear left
            ])

            # 3D model points
            model_points = np.float32([
                (5.0, 0.0, -52.0),  # 33 - nose bottom
                (0.0, -330.0, -65.0),  # 8 - chin
                (150.0, -150.0, -125.0),  # 54 - lip right
                (-150.0, -150.0, -125.0),  # 48 - lip left
                (250.0, -20.0, 40.0),  # 3 - ear right
                (-250.0, -20.0, 40.0),  # 13 - ear left
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

            # ONLY IN DEBUG MODE: Draw landmarks used head pose estimation
            self._draw_face_landmarks(shape=shape)

            # ONLY IN DEBUG MODE: Draw points used head pose estimation
            #self._draw_face_points(points=image_points)

            # ONLY IN DEBUG MODE: Write the euler angles on the frame
            self._draw_angles(pitch=pitch, yaw=yaw, roll=roll, left=start_x, top=start_y)

            # only need one face
            break

        return self.frame, pitch, yaw, roll

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

    def _draw_face_rect(self, start_y, start_x, end_x, end_y):
        """Draw a green rectangle (and text) around the face."""
        cv2.rectangle(self.frame, (start_x, start_y), (end_x, end_y),
                          (0, 0, 255), 1)

    def _draw_face_points(self, points):
        """Draw used points for head pose estimation"""
        if self.debug:
            for point in points:
                cv2.circle(self.frame, (point[0], point[1]), 3, (0, 255, 0), -1)

    def _draw_face_landmarks(self, shape):
        """Draw the landmarks used for head pose on the frame."""
        face = np.array([
            (shape.part(4).x, shape.part(4).y),
            (shape.part(10).x, shape.part(10).y),
            (shape.part(13).x, shape.part(13).y),
            (shape.part(24).x, shape.part(24).y),
            (shape.part(33).x, shape.part(33).y),
            (shape.part(34).x, shape.part(34).y),
            (shape.part(35).x, shape.part(35).y),
            (shape.part(0).x, shape.part(0).y),
            (shape.part(1).x, shape.part(1).y),
            (shape.part(2).x, shape.part(2).y),
            (shape.part(3).x, shape.part(3).y),
        ])

        nose = [
            (shape.part(5).x, shape.part(5).y),
            (shape.part(6).x, shape.part(6).y),
            (shape.part(7).x, shape.part(7).y),
            (shape.part(8).x, shape.part(8).y),
            (shape.part(9).x, shape.part(9).y),
        ]

        mouth = [
            (shape.part(18).x, shape.part(18).y),
            (shape.part(19).x, shape.part(19).y),
            (shape.part(20).x, shape.part(20).y),
            (shape.part(21).x, shape.part(21).y),
            (shape.part(22).x, shape.part(22).y),
            (shape.part(23).x, shape.part(23).y),
            (shape.part(25).x, shape.part(25).y),
            (shape.part(26).x, shape.part(26).y),
            (shape.part(27).x, shape.part(27).y),
            (shape.part(28).x, shape.part(28).y),
            (shape.part(29).x, shape.part(29).y),
        ]

        for (idx, (x, y)) in enumerate(face):
            if idx == len(face) - 1:
                break
            cv2.line(self.frame, (x, y), (face[idx + 1][0], face[idx + 1][1]), [0, 255, 0], 1)

        for (idx, (x, y)) in enumerate(nose):
            if idx == len(nose) - 1:
                break
            cv2.line(self.frame, (x, y), (nose[idx + 1][0], nose[idx + 1][1]), [0, 255, 0], 1)

        for (idx, (x, y)) in enumerate(mouth):
            if idx == len(mouth) - 1:
                break
            cv2.line(self.frame, (x, y), (mouth[idx + 1][0], mouth[idx + 1][1]), [0, 255, 0], 1)

        # coords = np.zeros((shape.num_parts, 2), dtype="int")
        #
        # for i in range(0, shape.num_parts):
        #     coords[i] = (shape.part(i).x, shape.part(i).y)
        #
        # for (i, (x, y)) in enumerate(coords):
        #     cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)
        #     cv2.putText(self.frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def _draw_angles(self, pitch, yaw, roll, left, top):
        """Write the euler angles on the frame"""
        if self.debug:
            label_x = left
            label_y = top - 3
            if label_y < 0:
                label_y = 0
        text = "Pitch: {0:.0f}, Yaw: {1:.0f}, Roll: {2:.0f}".format(pitch, yaw, roll)
        cv2.putText(self.frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


class CameraPoseCalculator(object):

    def __init__(self, debug, first_frame, second_frame, feature_detector):
        self.debug = debug
        self.first_frame = first_frame
        self.second_frame = second_frame
        self.feature_detector = feature_detector

    def calculate_camera_pose(self):
        # camera parameters
        d = np.array([-0.03432, 0.05332, -0.00347, 0.00106, 0.00000, 0.0, 0.0, 0.0]).reshape(1,
                                                                                             8)  # distortion coefficients
        K = np.array([1189.46, 0.0, 805.49, 0.0, 1191.78, 597.44, 0.0, 0.0, 1.0]).reshape(3, 3)  # Camera matrix
        K_inv = np.linalg.inv(K)

        # undistort the images first
        first_rect = cv2.undistort(self.first_frame, K, d)
        second_rect = cv2.undistort(self.second_frame, K, d)

        # extract key points and descriptors from both images
        first_key_points, first_descriptors = self.feature_detector.detectAndCompute(first_rect, None)
        second_key_points, second_descriptos = self.feature_detector.detectAndCompute(second_rect, None)

        # match descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        matches = matcher.match(first_descriptors, second_descriptos)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # generate lists of point correspondences
        first_match_points = np.zeros((len(matches), 2), dtype=np.float32)
        second_match_points = np.zeros_like(first_match_points)
        for i in range(len(matches)):
            first_match_points[i] = first_key_points[matches[i].queryIdx].pt
            second_match_points[i] = second_key_points[matches[i].trainIdx].pt

        # estimate fundamental matrix
        F, mask = cv2.findFundamentalMat(first_match_points, second_match_points, cv2.FM_RANSAC, 0.1, 0.99)

        # decompose into the essential matrix
        E = K.T.dot(F).dot(K)

        # decompose essential matrix into R, t
        U, S, Vt = np.linalg.svd(E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(mask)):
            if mask[i]:
                # normalize and homogenize the image coordinates
                first_inliers.append(K_inv.dot([first_match_points[i][0], first_match_points[i][1], 1.0]))
                second_inliers.append(K_inv.dot([second_match_points[i][0], second_match_points[i][1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in front of both cameras
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self.in_front_of_both_cameras(first_inliers, second_inliers, R, T):

            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
            if not self.in_front_of_both_cameras(first_inliers, second_inliers, R, T):

                # Third choice: R = U * Wt * Vt, T = u_3
                R = U.dot(W.T).dot(Vt)
                T = U[:, 2]

                if not self.in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                    # Fourth choice: R = U * Wt * Vt, T = -u_3
                    T = - U[:, 2]

        pitch, yaw, roll = calculate_euler_angles(R, inverse=True)
        return self.first_frame, self.second_frame, pitch, yaw, roll

    @staticmethod
    def in_front_of_both_cameras(first_points, second_points, rot, trans):
        """Checks if the point correspondences are in front of both images"""
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                                                                second)
            first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True
