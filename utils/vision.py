from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary
import cv2.aruco as aruco


class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config
    
    # function to convert rvec, tvec to transform matrix
    @staticmethod
    def to_tf(rvec, tvec, order="xyz"):
        tf = np.identity(4, dtype=float)
        r = Rotation.from_euler(order,rvec,degrees=False)
        rot_matrix = r.as_matrix()
        tf[:3, :3] = rot_matrix
        tf[:3, 3] = tvec
        return tf

    def detections(self, img: np.ndarray, draw_img:np.ndarray, robot_pose: tuple, kind: str = "aruco") -> tuple:
        #ids, landmark_rs, landmark_alphas, landmark_positions = [222], [1.70], [2.], [[2,1]]
        ids, landmark_rs, landmark_alphas, landmark_positions = [], [], [], []
        if kind == "aruco":
            # Extract rvec and tvec from robot_pose
            id, rvec, tvec = robot_pose
            ids.extend(id)

            # finding distance aurco from robot
            distance = np.linalg.norm(tvec)
            landmark_rs.extend(distance)

            # Calculate angles to landmarks
            rot_matrix, _ = cv2.Rodrigues(rvec)   # changing into  rotation matrix
            # Convert rotation matrix to Euler angles in degrees
            r = Rotation.from_matrix(rot_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)

            # Extract individual Euler angles (in degrees)
            yaw, pitch, roll = euler_angles

            # Yaw is the rotation around the vertical axis (usually the y-axis)/left or right rotation
            # Pitch is the rotation around the lateral axis (usually the x-axis)/ up or down tilt of an object
            # Roll is the rotation around the longitudinal axis (usually the z-axis)/ rotation from side to side
            # Assuming the roataion in y-axis
            landmark_alphas.extend(yaw)
            
            # aruco marker transform w.r.t. camera
            rot_matrix, _ = cv2.Rodrigues(np.array(rvec))

            aruco_tf = np.identity(4, dtype=float)
            aruco_tf[:3, :3] = rot_matrix
            aruco_tf[:3, 3] = tvec

            # Print the aruco_tf
            print("aruco_tf:")
            print(aruco_tf)

            # camera transform w.r.t. robot base
            rvec = [np.radians(-90), 0, np.radians(-135)]
            tvec = [0, 0.03, 0.20]  # in meters
            camera_tf = Vision.to_tf(rvec, tvec, order="ZYX")

            # Print the camera_tf
            print("camera_tf:")
            print(camera_tf)

            # aruco marker transform w.r.t. robot base
            aruco_robot_tf = np.dot(camera_tf, aruco_tf)

            # Print the aruco_robot_tf
            print("aruco_robot_tf:")
            print(aruco_robot_tf)

            # we can read out x, y, z coordinates of aruco marker transform w.r.t. robot base
            print(f"x = {aruco_robot_tf[0,3]}")
            print(f"y = {aruco_robot_tf[1,3]}")
            landmark_positions = landmark_positions.append(aruco_robot_tf[0,3], aruco_robot_tf[1,3])

        return ids, landmark_rs, landmark_alphas, landmark_positions

