import numpy as np
from utils import *
from scipy.linalg import expm, inv
from hat import hat

###########################################################################
######################### D. VISUAL-INERTIAL SLAM #########################
###########################################################################

if __name__ == '__main__':

	####################################################################
	######################### PART 0. GET DATA #########################
	####################################################################

	# only choose ONE of the following data
	# data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part 
	filename = "./data/10.npz"
	t,features_all,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
	#print('t:', t) # 1,3026
	#print('features_all:', features_all) # 4,13289,3026 # coordinates,number of features,timestamps
	#print('linear_velocity:', linear_velocity) # 3,3026
	#print('angular_velocity:', angular_velocity) #3,3026
	#print('K:', K) # 3,3
	#print(b) # 1,1
	#print(imu_T_cam) #4,4
	# get number of features
	num_features_all = np.shape(features_all)[1]  # 13289
	# get number of timestamps
	num_time = np.shape(features_all)[2]  # 3026

	# data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra credit 
	#filename = "./data/03.npz"
	#t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	# Obtained Above
	# t = timestamps in UNIX standard seconds since epoch 1/1/1970
	# features = detected visual features
	# linear_velocity = lienar velocity in imu frame
	# angular_velocity = angular velocity in imu frame
	# K (instrinsic calibration) = camera calibration matrix (K = [f*s_u  0 c_c; 0 f*s_v c_v; 0 0 1])
	# b = stereo baseline (meters)
	# cam_T_imu (extrinsic calibration) = transformation in SE(3) from left camera to imu frame

	###############################################################################
	######################### PART 0 - DATA PREPROCESSING #########################
	###############################################################################

	# reshape linear/angular velocity into needed nx3 dimensions
	linear_velocity = np.transpose(linear_velocity)  # 3026,3
	angular_velocity = np.transpose(angular_velocity)  # 3026,3

	# store every 5 features
	landmarks_every_five = features_all[:, 1:num_features_all:5, :]  # 4,2658,3026
	# get number of features after taking every 5
	num_t_sampled_features = np.shape(landmarks_every_five)[2]  # 3026

	############################################################################################
	######################### PART 0 - INITIALIZE VARIABLES/PARAMETERS #########################
	############################################################################################

	#initialize imu mean/covariance storage variables
	mu_t_t = np.identity(4)
	sigma_t_t = np.identity(6)

	#initialize imu pose ot mean position (identity matrix)
	pose_size = (4, 4, num_t_sampled_features) #4x4x3026
	pose = np.zeros(pose_size) #pose initialized with all zeros
	pose[:,:,0] = mu_t_t #pose at timestmap 0 initialized with identity matrix

	#standard deviation for gaussian measurement noise
	V = 95
	#get number of sampled landmarks
	num_landmarks_every_five = np.shape(landmarks_every_five)[1] #2658
	# initialize world frame landmark pose and covariance matrix
	mu_t_landmarks = -1 * np.ones((4,num_landmarks_every_five)) #initialize landmarks as empty
	sigma_t_landmarks = np.identity(3*num_landmarks_every_five) * V

	initialization = np.vstack((np.identity(3), np.array([0,0,0])))
	I_num_landmarks_every_five = np.identity(num_landmarks_every_five)
	initialization = np.kron(I_num_landmarks_every_five,initialization)

	# get stereo calibration calibration matrix M - Lecture 13, Slide 2
	# get camera intrinsic parameters from K
	f_su, f_sv, cu, cv = K[0][0], K[1][1], K[0][2], K[1][2]
	# calculate stereo calibration matrix
	M = np.array([[f_su, 0, cu, 0],
				  [0, f_sv, cv, 0],
				  [f_su, 0, cu, -f_su * b],
				  [0, f_sv, cv, 0]])
	# f = focal length [m]
	# su, sv = pixel scaling [pixles/m]
	# cu, cv = principal point [pixels]
	# b = stereo baseline [m]

	#iterate through landmarkfeatures (every 5 sampled)
	for i in range(num_t_sampled_features): #0 to 3026
		#for viewing code-running progress
		print('i:',i,'/',num_t_sampled_features)
		# calculate time difference between current/previous timestamp
		tau = t[0, i] - t[0, i - 1]

		################################################################################################
		######################### PART A - IMU LOCALIZATION VIA EKF PREDICtION #########################
		################################################################################################

		#get linear/angular velocity
		velocity_t = linear_velocity[i,:] #3,1
		angular_velocity_t = angular_velocity[i,:] #3,1

		#calculate hat linear/angular velocity matrices - lecture 13, slide 15
		velocity_t_hat = hat(velocity_t) #3,3
		angular_velocity_t_hat = hat(angular_velocity_t) #3,3
		#calculate u_t matrix - lecture 13, slide 15
		control_t = np.array([[velocity_t[0], velocity_t[1], velocity_t[2]], [angular_velocity_t[0], angular_velocity_t[1], angular_velocity_t[2]]])
		#calculate u_t_hat matrix - lecture 13, slide 15
		control_t_hat = np.array([[angular_velocity_t_hat[0][0], angular_velocity_t_hat[0][1], angular_velocity_t_hat[0][2], velocity_t[0]],
								[angular_velocity_t_hat[1][0], angular_velocity_t_hat[1][1], angular_velocity_t_hat[1][2], velocity_t[1]],
								[angular_velocity_t_hat[2][0], angular_velocity_t_hat[2][1], angular_velocity_t_hat[2][2], velocity_t[2]],
								[0, 0, 0, 0]])
		#calculate u_t_hat_skinny matrix - lecture 13, slide 15
		control_t_hat_thin = np.array([[angular_velocity_t_hat[0][0], angular_velocity_t_hat[0][1], angular_velocity_t_hat[0][2], velocity_t_hat[0][0], velocity_t_hat[0][1], velocity_t_hat[0][2]],
									   [angular_velocity_t_hat[1][0], angular_velocity_t_hat[1][1], angular_velocity_t_hat[1][2], velocity_t_hat[1][0], velocity_t_hat[1][1], velocity_t_hat[1][2]],
									   [angular_velocity_t_hat[2][0], angular_velocity_t_hat[2][1], angular_velocity_t_hat[2][2], velocity_t_hat[2][0], velocity_t_hat[2][1], velocity_t_hat[2][2]],
									   [0, 0, 0, angular_velocity_t_hat[0][0], angular_velocity_t_hat[0][1], angular_velocity_t_hat[0][2]],
									   [0, 0, 0, angular_velocity_t_hat[1][0], angular_velocity_t_hat[1][1], angular_velocity_t_hat[1][2]],
									   [0, 0, 0, angular_velocity_t_hat[2][0], angular_velocity_t_hat[2][1], angular_velocity_t_hat[2][2]]])

		#predict new pose mean and covariance - lecture 13, slide 15
		#EKF predict new pose mean
		mu_new_t = np.dot( expm(-tau*control_t_hat), mu_t_t )
		# EKF predict new pose covariance
		sigma_exponential = expm(-tau*control_t_hat_thin)

		# calculate noise matrix - lecture 13, slide 15
		noise = np.diag(np.random.normal(0, 1, 6))
		W = tau*tau*noise
		sigma_sub_t_plus_one_t = np.dot( np.dot(sigma_exponential,sigma_t_t), np.transpose(sigma_exponential) ) + W

		#########################################################################################
		######################### D. VISUAL-INERTIAL SLAM  - imu update #########################
		#########################################################################################

		#update mean/covariance of imu pose
		mu_t_t = mu_new_t
		sigma_t_t = sigma_sub_t_plus_one_t

		# update imu with new predcited pose mean and covariance
		pose[:, :, i] = inv(mu_new_t)

		##########################################################################################
		######################### PART B - FEATURE DETECTION AND MAPPING #########################
		##########################################################################################

		# *skipping
		# (optional)

		############################################################################################
		######################### PART C - LANDMARK MAPPING VIA EKF UPDATE #########################
		############################################################################################

		'''
		# calculate camera frame to world frame using predicted pose from Part B
		c_T_w_transform = np.dot(imu_T_cam, mu_t_t)
		# calculate world frame to camera frame transform
		w_T_c_transform = inv(c_T_w_transform)
		'''

		# calculate camera frame to world frame using predicted pose from Part B
		c_T_w_transform = np.dot(inv(imu_T_cam), mu_t_t)
		# calculate world frame to camera frame transform
		w_T_c_transform = inv(c_T_w_transform)

		# get all landmark features at current timestamp
		features_i = landmarks_every_five[:, :, i]

		#find non-empty features and store (non-empty features have values [-1, -1, -1, -1])
		# sum vector of each poes of each landmark feature
		sum_of_feature_vectors = np.sum(features_i[:, :], axis=0)
		# test if landmark feature vector sums a nonempty (!=-4) or not
		non_empty_features_idx = np.array(np.where(sum_of_feature_vectors != -4))
		# get number of non-empty landmark features
		num_non_empty_features_idx = np.size(non_empty_features_idx)

		#initialize feature and feature index storage variables
		features_updated = np.zeros((4, 0))  # initially empty
		idx_landmark_features_updated = np.zeros((0, 0), dtype=np.int8) #initially empty
		#check if there are any non-empty landmark features for current ith iteration
		if (num_non_empty_features_idx > 0):
			#get all non-empty landmark features and store them
			nonempty_landmark_features_i = features_i[:, non_empty_features_idx]
			nonempty_landmark_features_coord = nonempty_landmark_features_i.reshape(4, num_non_empty_features_idx)

			#get feature landmarks from nonempty features
			landmark_features = np.ones((4, np.shape(nonempty_landmark_features_coord)[1]))
			landmark_features[0, :] = (nonempty_landmark_features_coord[0, :]
									   - cu)*b / (nonempty_landmark_features_coord[0, :]-
												  nonempty_landmark_features_coord[2, :])
			landmark_features[1, :] = (nonempty_landmark_features_coord[1, :] - cv) * \
									  (-M[2, 3]) / (M[1, 1]*(nonempty_landmark_features_coord[0, :]-
															 nonempty_landmark_features_coord[2, :]))
			landmark_features[2, :] = -(-f_su * b) / (nonempty_landmark_features_coord[0, :]-
													  nonempty_landmark_features_coord[2, :])
			#convert landmark features from world frame to camera frame
			landmark_features = np.dot(w_T_c_transform, landmark_features)

			# ESTIMATE COORDINATES OF LANDMARKS THAT GENERATED OBSERVED FEATURES
			#iterate through nonempty features found above
			for j in range(num_non_empty_features_idx):

				m_j = non_empty_features_idx[0, j]
				#if landmark seen before
				if (np.array_equal(mu_t_landmarks[:, m_j], [-1, -1, -1, -1])):
					mu_t_landmarks[:, m_j] = landmark_features[:, j]
				#if landmark not seen before
				else:
					# reshape updated features matrix
					features_updated = np.hstack((features_updated, landmark_features[:, j].reshape(4, 1)))
					#append landmark features to stored features
					idx_landmark_features_updated = np.append(idx_landmark_features_updated, m_j)

			#check if there are updated features
			length_updated_landmark_features = len(idx_landmark_features_updated)
			if (length_updated_landmark_features != 0):

				#reshape landmark matrix variable to needed form
				mu_landmark_shape = (4, length_updated_landmark_features)
				mu_landmark = mu_t_landmarks[:, idx_landmark_features_updated].reshape(mu_landmark_shape)

				#Initialize projection matrix and Jacobian matrix needed for equation calculations
				#initialize projection matrix
				P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
				P_transposed = np.transpose(P)
				# calculate the Jacobian
				total_number_of_features = np.shape(landmarks_every_five)[1]

				for n in range(length_updated_landmark_features):
					#calculate projection function - Lecture 13, Slide 5
					pi_q = np.dot(c_T_w_transform, mu_landmark[:, n])
					#calculate derivative of projection function - Lecture 13, Slide 5
					dpi_over_dq = (np.array([[1, 0, -pi_q[0]/pi_q[2], 0],[0, 1, -pi_q[1]/pi_q[2], 0],[0, 0, 0, 0],[0, 0, -pi_q[3]/pi_q[2], 1]])) / pi_q[2]

					#initialize observation model Jacobian matrix - Lecture 13, Slide 6
					size_idx_landmark_features_updated = np.size(idx_landmark_features_updated)
					H_t_plus_one_i_j = np.zeros((4*size_idx_landmark_features_updated, 3*total_number_of_features))
					#calculate bservation model Jacobian matrix - Lecture 13, Slide 8
					H_t_plus_one_i_j[4*n : 4*(n+1), 3*m_j : 3*(m_j+1)] = np.dot(np.dot(np.dot(M, dpi_over_dq), c_T_w_transform), P_transposed)

				#calculate K - Lecture 13, Slide 8
				K_t = np.dot(np.dot(sigma_t_landmarks, np.transpose(H_t_plus_one_i_j)), inv(np.dot(np.dot(H_t_plus_one_i_j, sigma_t_landmarks), np.transpose(H_t_plus_one_i_j)) + np.identity(4 * length_updated_landmark_features) * V))

				#calculate landmark coordinates in world frame
				pi_q_2 = np.dot(c_T_w_transform, mu_landmark)
				#calculate projection function
				projection_function = pi_q_2 / pi_q_2[2, :]
				#calculate z_t
				z_t = np.dot(M, projection_function)

				#calculate updated landmark pose (mean/covariance) - Lecture 13, Slide 8
				# reshape stored current feature
				z = features_i[:, idx_landmark_features_updated].reshape((4, length_updated_landmark_features))
				mu_t_landmarks = (mu_t_landmarks.reshape(-1, 1, order='F') + np.dot(np.dot(initialization, K_t), (z - z_t).reshape(-1, 1, order='F'))).reshape(4, -1, order='F')
				sigma_t_landmarks = np.dot((np.identity(3 * np.shape(landmarks_every_five)[1]) - np.dot(K_t, H_t_plus_one_i_j)), sigma_t_landmarks)

	#########################################################################################################################
	######################### D. VISUAL-INERTIAL SLAM  - combine imu prediction and landmark update #########################
	#########################################################################################################################
	#visualize vehicle trajectory + landmarks using function from utils.py
		if i%1000==0:
			visualize_trajectory_2d(pose, mu_t_landmarks, show_ori=True)
	visualize_trajectory_2d(pose, mu_t_landmarks, show_ori=True)











