##############################################################################
# This python file is ex2 in image processing course.
# the next script is functions for Fourier transformation on vectors and matrix
# and  audio  manipulations using different sample frequencies and Fourier
# transform. and editing their grayscale deviation for different needs.
##############################################################################
import numpy as np
import scipy.io.wavfile as siw
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
from skimage.color import rgb2gray

MAX_SEGMENT = 255

CHANGE_RATE_FILE = "change_rate.wav"

CHANGE_SAMPLE_FILE = "change_samples.wav"


def DFT(signal):
    """
    Discrete Fourier Transform on array.
    :param signal: 1d Numpy array dtype float 64 shape (N,).
    :return: complex128 shape (N,)
    """
    N = signal.shape[0]
    if N == 0:
        return signal

    i, j = np.meshgrid(np.arange(N), np.arange(N))
    fourier_matrix = np.exp((-2 * np.pi * 1J * i * j) / N)
    fourier_signal = np.dot(fourier_matrix, signal)
    return fourier_signal


def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform on array.
    :param fourier_signal: complex128 shape (N,)
    :return: 1d Numpy array complex128 shape (N,).
    """
    N = fourier_signal.shape[0]
    if N == 0:
        return fourier_signal

    i, j = np.meshgrid(np.arange(N), np.arange(N))
    inverse_fourier_matrix = np.exp((2.0 * np.pi * 1J * i * j) / N)
    signal = 1 / N * (np.dot(inverse_fourier_matrix, fourier_signal))
    return np.real_if_close(signal)


def DFT2(image):
    """
    Discrete Fourier Transform on a matrix.
    :param image:2d Numpy array dtype float 64 shape (N,M).
    :return: 2d Numpy array complex128 shape (N,M)
    """
    return (DFT(DFT(image).T)).T


def IDFT2(fourier_image):
    """
    Inverse Discrete Fourier Transform on a matrix.
    :param fourier_image:  2d Numpy array complex128 shape (N,M)
    :return:  2d Numpy array complex128 shape (N,M)
    """
    return (IDFT(IDFT(fourier_image).T)).T


def change_rate(filename, ratio):
    """
    Change the duration of an audio by changing rate of sampling.
    :param filename: path for wav file.
    :param ratio: positive float64 represent duration change.
    """
    ratio_orig, audio = siw.read(filename)
    new_ratio = ratio_orig * ratio
    siw.write(CHANGE_RATE_FILE, int(round(new_ratio)), audio)


def change_samples(filename, ratio):
    """
    Change the speed of an audio by changing the sampling ratio using fourier.
    :param filename: path for wav file.
    :param ratio: positive float64 represent sampled change.
    :return: 1D ndarray of dtype float64
    """
    ratio_orig, audio = siw.read(filename)
    data_after_resize = resize(audio, ratio).astype(np.float64)
    siw.write(CHANGE_SAMPLE_FILE, int(ratio_orig),
              data_after_resize.astype(np.int16))
    return data_after_resize


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
    samples_change = int(abs(N - new_samples))
    start = samples_change // 2
    end = N - (samples_change - start)
    if samples_change == N and ratio > 1:
        return np.empty(shape=(0,))
    elif ratio > 1:
        new_f_w = f_w[start:end]
    elif ratio < 1:
        new_f_w = np.zeros(int(new_samples)).astype(np.complex128)
        end = int(N + start)
        new_f_w[start:end] = f_w
    else:
        new_f_w = f_w
    new_data = IDFT(np.fft.ifftshift(new_f_w))
    return new_data


def resize_spectrogram(data, ratio):
    """
    This function speed up/slow down a wav file, without changing the pitch.
    :param data: a sample of the wav file dtype float64
    :param ratio: the ratio we want our audio to be changed by.
    :return: the data after speeding/slowing.
    """
    spectrogram = stft(data)
    new_spectrogram = np.apply_along_axis(resize, 1, spectrogram, ratio)
    return istft(new_spectrogram)


def resize_vocoder(data, ratio):
    """
    This function speed up/slow down a wav file, without changing the pitch.
    while re-arrange the phase.
    :param data: a sample of the wav file dtype float64
    :param ratio: the ratio we want our audio to be changed by.
    :return: the data after speeding/slowing.
    :param data:
    :param ratio:
    :return:
    """
    stft_data = stft(data)
    phase_correct_spectrogram = phase_vocoder(stft_data, ratio)
    return istft(phase_correct_spectrogram)


def conv_der(im):
    """
    This function calculate derivative of a photo using convolution.
    :param im: matrix represent gray scale of a photo.
    :return: matrix represent magnitude of a derivative photo.
    """
    line_vector_conv = np.array([[0.5, 0, -0.5]]).transpose()
    column_vector_conv = np.array([[0.5, 0, -0.5]])
    return np.sqrt(np.abs(signal.convolve2d(im, line_vector_conv)) ** 2 +
                   np.abs(signal.convolve2d(im, column_vector_conv)) ** 2)


def fourier_der(im):
    """
    This function calculate derivative of a photo using fourier.
    :param im: matrix represent gray scale of a photo.
    :return: matrix represent magnitude of a derivative photo.
    """
    number_of_lines = im.shape[0]
    number_of_columns = im.shape[1]
    u_frequencies_vector = np.arange(-number_of_columns // 2,
                                     number_of_columns // 2, dtype=np.float64)
    v_frequencies_vector = np.arange(-number_of_lines // 2,
                                     number_of_lines // 2, dtype=np.float64)
    derivative_x_image = \
        (2 * np.pi / number_of_columns) * IDFT2(
            (DFT2(im).np.fft.fftshift * u_frequencies_vector).np.fft.ifftshift)
    derivative_y_image = \
        (2 * np.pi / number_of_lines) * IDFT2(
            (DFT2(im).np.fft.fftshift * v_frequencies_vector).np.fft.ifftshift)
    derivative_image = np.sqrt(np.abs(derivative_x_image) ** 2 +
                               np.abs(derivative_y_image) ** 2)
    return derivative_image


# change_rate(
#     "C:/Users/Roy\PycharmProjects/ex2-royschossberge/external/aria_4kHz.wav",
#     2)
# change_samples(
#     "C:/Users/Roy\PycharmProjects/ex2-royschossberge/external/aria_4kHz.wav",
#     2)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def read_image(filename, representation):
    """
    The next lines preform a image read to a matrix of numpy.float64 using
    imagio and numpy libraries.
    :param filename: a path to jpg image we would like to read.
    :param representation: 1 stands for grayscale , 2 for RGB.
    :return: image_mat - a numpy array represents the photo as described above.
    """
    image = imread(filename)
    if representation == 1:
        image_mat = np.array(rgb2gray(image))
    else:
        image_mat = np.array(image.astype(np.float64))
        image_mat /= MAX_SEGMENT
    return image_mat

# ratio_orig, audio = siw.read("C:/Users/Roy\PycharmProjects/ex2-royschossberge/external/beautiful_Voice.wav")
# siw.write("samplevoco.wav", ratio_orig, resize_vocoder(audio,2).astype(np.int16))
# siw.write("samplesp.wav", ratio_orig, resize_spectrogram(audio,2).astype(np.int16))
