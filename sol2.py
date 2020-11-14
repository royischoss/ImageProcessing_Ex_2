##############################################################################
# This python file is ex2 in image processing course.
# the next script is functions for Fourier transformation on vectors and matrix
# and  audio  manipulations using different sample frequencies and Fourier
# transform. and editing their grayscale deviation for different needs.
##############################################################################
import numpy as np
import scipy.io.wavfile as siw



def DFT(signal):
    """
    Discrete Fourier Transform on array.
    :param signal: 1d Numpy array dtype float 64 shape (N,).
    :return: complex128 shape (N,)
    """
    N = signal.shape[0]
    w = np.exp(-2 * np.pi * 1J / N)
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    fourier_matrix = np.power(w, i * j)
    fourier_signal = (np.dot(fourier_matrix, signal)).astype(np.complex128)
    return fourier_signal


def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform on array.
    :param fourier_signal: complex128 shape (N,)
    :return: 1d Numpy array complex128 shape (N,).
    """
    N = fourier_signal.shape[0]
    w = np.exp(2 * np.pi * 1J / N)
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    inverse_fourier_matrix = np.power(w, i * j)
    fourier_signal = (np.dot(inverse_fourier_matrix,
                             fourier_signal) * (1 / N)).astype(np.complex128)
    return np.real(fourier_signal)


def DFT2(image):
    """
    Discrete Fourier Transform on a matrix.
    :param image:2d Numpy array dtype float 64 shape (N,M).
    :return: 2d Numpy array complex128 shape (N,M)
    """
    return np.apply_along_axis(DFT, 0, np.apply_along_axis(DFT, 1, image))


def IDFT2(fourier_image):
    """
    Inverse Discrete Fourier Transform on a matrix.
    :param fourier_image:  2d Numpy array complex128 shape (N,M)
    :return:  2d Numpy array complex128 shape (N,M)
    """
    return np.apply_along_axis(IDFT, 0, np.apply_along_axis(IDFT, 1,
                                                            fourier_image))


def change_rate(filename, ratio):
    """
    Change the duration of an audio by changing rate of sampling.
    :param filename: path for wav file.
    :param ratio: positive float64 represent duration change.
    """
    ratio_orig, audio = siw.read(filename)
    new_ratio = ratio_orig * ratio
    siw.write("change_rate.wav", new_ratio, audio)


def change_samples(filename, ratio):
    """
    Change the speed of an audio by changing the sampling amount using fourier.
    :param filename: path for wav file.
    :param ratio: positive float64 represent sampled change.
    :return: 1D ndarray of dtype float64
    """
    ratio_orig, audio = siw.read(filename)
    siw.write("change_samples.wav", ratio_orig * ratio, resize(audio, ratio))


def resize(data, ratio):
    """
    resizing the audio 1d array to the new samples ratio using fourier.
    :param data: 1d array of dtype 64float.
    :param ratio: the ratio we want to change to.
    :return: 1d data array dtype complex 128 in the right length.
    """
    f_w = DFT(data)
    f_w = np.fft.fftshift(f_w)
    N = f_w.shape[0]
    new_samples = N // ratio
    if ratio > 1:
        new_f_w = f_w[int(round(-new_samples / 2)): int(round(new_samples / 2))]
    else:
        new_f_w = np.zeros(new_samples)
        new_f_w[round(-N / 2): round(N / 2)] = f_w[round(-N / 2): round(N / 2)]
    new_data = IDFT(np.fft.ifftshift(new_f_w))
    return new_data


change_rate("C:/Users/Roy\PycharmProjects/ex2-royschossberge/external/aria_4kHz.wav", 2)
change_samples("C:/Users/Roy\PycharmProjects/ex2-royschossberge/external/aria_4kHz.wav", 2)




