# Does hat operation on vectors
import numpy as np

def hat(vector_input):
	vector_hat_out = np.array([[0, -vector_input[2], vector_input[1]],
					       [vector_input[2], 0, -vector_input[0]],
					       [-vector_input[1], vector_input[0], 0]])
	return vector_hat_out