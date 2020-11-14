##############################################################################
# This python file is ex2 in image processing course.
# the next script is functions for Fourier transformation on vectors and matrix
# and  audio  manipulations using different sample frequencies and Fourier
# transform. and editing their grayscale deviation for different needs.
##############################################################################
import numpy as np



def DFT(signal):
    """
    Discrete Fourier Transform on array.
    :param signal: 1d Numpy array dtype flot 64 shape (N,).
    :return:
    """
    N = signal.shape[0]
    w = np.exp(-2 * np.pi * 1J / N)
    r = np.arange(N)
    fourier_matrix = np.vander(w, **r, increasing=True)
    fourier_signal = (np.dot(fourier_matrix, signal)).astype(np.complex128)
    return fourier_signal


def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform on array.
    :param fourier_signal:
    :return:
    """
    N = fourier_signal.shape[0]
    w = np.exp(2 * np.pi * 1J / N)
    r = np.arange(N)
    fourier_matrix = np.vander(w, **r, increasing=True)
    fourier_signal = (np.dot(fourier_matrix,
                             fourier_signal) * (1 / N)).astype(np.complex128)
    return np.real(fourier_signal)


def DFT2(image):
    """
    Discrete Fourier Transform on a matrix.
    :param image:
    :return:
    """
    return np.apply_along_axis(DFT, 0, np.apply_along_axis(DFT, 1, image))


def IDFT2(fourier_image):
    """
    Inverse Discrete Fourier Transform on a matrix.
    :param fourier_image:
    :return:
    """
    return np.apply_along_axis(IDFT, 0, np.apply_along_axis(IDFT, 1,
                                                            fourier_image))

def change_rate(filename, ratio):
    """
    Change the duration of an audio by changing rate of sampling.
    :param filename:
    :param ratio:
    :return:
    """






