# Visual-Inertial-SLAM

DESCRIPTION OF FILES
hw3_main.py
- does imu localization via EKF prediction step
- does landmark mapping via EKF update step
- combines EKF prediction/update and landmark mapping to perform
  visual-inertial SLAM
- produces map with tracked vehicle pose and sensed landmarks

hat.py
- hat() - function which does hat operation on a vector

utils.py
- load_data() - used to load timestamps, features, linear velocity,
                angular velocity, (left)camera intrinsic matrix,
                stereo camera baseline, imu to camera frame transform
- visualize_trajectory_2d() - used to visualize vehicle pose and 
	        sensed landmarks
