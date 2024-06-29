import numpy as np
import cmath

# Functions, which behave as the correspondent matlab functions.

def repmat(input_matrix, reps):
    len_reps = len(reps)
    len_input = len(input_matrix.shape)

    if len_input < len_reps:
        input_matrix_append = input_matrix
        while len(input_matrix_append.shape) < len_reps:
            input_matrix_append = np.reshape(input_matrix, (input_matrix_append.shape + (1,)))

        output_matrix = np.tile(input_matrix_append, reps)

    elif len_input == len_reps:
        output_matrix = np.tile(input_matrix, reps)

    elif len_input > len_reps:
        reps_append = reps
        while len(reps_append) < len_input:
            reps_append = reps_append + (1,)

        output_matrix = np.tile(input_matrix, reps_append)

    return output_matrix

def acos(inArray):

    acos = np.zeros(inArray.shape, dtype=complex)

    for index, _ in np.ndenumerate(inArray):
        acos[index] = cmath.acos(inArray[index])

    outArray = acos

    return outArray

