# CV_Robotic_project

make changes in vision.py detection function which return (** still have to change the alphas/angle which angle to take: for now took the yaw, and the camera w.r.t robot has to change)

1) landmark_ids: shape (m,)
2) landmark_rs: shape (m,) - distance to landmark
3) landmark_alphas: shape (m,) - angle to landmark
4) landmark_positions: shape (m, 2) - list of (x, y) -> aruco marker position

Next step: figure out how to call this function from publisher, and SLAM. 
