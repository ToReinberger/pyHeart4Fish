"""
pyHeart4Fish bright field module

Author: Dr. Tobias Reinberger and Viviana Vedder
Data: 07.03.2023
Version: beta 0.0.1

"""

from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
import os
import argparse
from scipy.fft import rfft, rfftfreq
from aicsimageio.readers import CziReader
import czifile
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy import stats
import cv2
import io


class ExtractImages:
    """
    * takes the input path and stores frames from movies (czi, avi, mp4) or single images from folder (tif, jpg, png)
      in arrays as numpy arrays
    * for avi and czi frames per second and other meta data are extracted from the original data
    """
    def __init__(self, path_in):
        self.path_in = path_in

    def extract_images_from_avi(self):
        """
        returns every second image as array
        :param path_in:
        :return: avi_images
        """
        print("\nextract images from avi for ", self.path_in)
        avi_images = []
        avi_images_org = []
        vidcap = cv2.VideoCapture(self.path_in)

        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        # print("um per px: ", vidcap.get(2))
        fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)     # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count/fps)
        print("\n ### Frame parameter ###")
        print("Height (px): ", vidcap.get(4))
        print("Width (px): ", vidcap.get(3))
        print("frames per sec: ", fps)
        print("duration (sec): ", (frame_count/fps))
        print("Total number of images: ", int(vidcap.get(7) - 1))
        success, image_from_avi = vidcap.read()

        count = 0
        while success:
            # vidcap.set(cv2.CAP_PROP_POS_MSEC, count)    # added this line
            success, image_from_avi = vidcap.read()
            if success:
                count = count + 1
                print(count, end="\r")
                # print(len(image_from_avi), len(image_from_avi[0]), len(image_from_avi[0][0]))
                gray = cv2.cvtColor(image_from_avi, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                # make_borders_to_image
                top, bottom, left, right = [int(width / 8)] * 4
                gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                image_from_avi_temp = cv2.copyMakeBorder(image_from_avi, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                         value=0)
                avi_images_org.append(image_from_avi_temp)
                avi_images.append(gray)
                # plt.imshow(gray)
                # plt.show()
        if skip_images_factor > 1:
            return avi_images[::skip_images_factor], avi_images_org[::skip_images_factor], fps / skip_images_factor
        else:
            return avi_images, avi_images_org, fps

    def extract_images_from_czi(self):
        # reader = CziReader(self.path_in)  > CziReader can not deal with bright field images of this kind
        # images_temp = reader.data
        images_temp = czifile.imread(self.path_in)
        init_shape = images_temp[0].shape
        if len(init_shape) > 3:
            images_temp = [np.reshape(img_temp, (init_shape[1:])) for img_temp in images_temp]
        images_temp = [cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2GRAY) for x in images_temp]
        images_org = images_temp.copy()
        print(len(images_temp))
        # print(images_temp)
        """for img_ in images_temp:
            plt.imshow(img_)
            plt.show()"""
        # w, h = reader.dims['X'][0], reader.dims['Y'][0]

        # get_frame_rate_from_meta
        aqu_time = 24
        # a = czifile.CziFile((self.path_in).metadata()
        c = czifile.CziFile(self.path_in).subblocks()
        frame_time_stamp = []
        for frame in c:  # iterate through every frame
            # print(elem)
            frame = str(frame)
            aqu_time = frame[frame.find("AcquisitionTime") + len("AcquisitionTime: "): frame.find("DetectorState") - 5].strip()
            aqu_time = aqu_time.replace(",", "").replace("'", "").split("T")[1][:-1]
            aqu_time = aqu_time.split(":")
            h = float(aqu_time[0])
            m = float(aqu_time[1])
            s = float(aqu_time[2])
            time_in_sec = h * 60 * 60 + m * 60 + s

            frame_time_stamp.append(time_in_sec)
        acq_time = frame_time_stamp[-1] - frame_time_stamp[0]

        if acq_time == 0:
            print("!!!!!!!!!!!!!!!!! ERROR in", self.path_in)
            frames_per_sec = 9.5
        else:
            frames_per_sec = round(len(frame_time_stamp) / acq_time, 1)
            print("Acquisition time (s):", acq_time,
                  "| Frames:", len(frame_time_stamp),
                  "| Frame rate (frames/s):", frames_per_sec)

        if skip_images_factor > 1:
            return images_temp[::skip_images_factor], images_org[::skip_images_factor], frames_per_sec / skip_images_factor
        else:
            return images_temp, images_org, frames_per_sec

    def extract_images_from_mp4(self):
        """
        returns every second image as array
        :param path_in:
        :return: frames as array
        """
        print("\nextract images from mp4 for ", self.path_in)
        avi_images = []
        avi_images_org = []
        vidcap = cv2.VideoCapture(self.path_in)

        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        # print("um per px: ", vidcap.get(2))
        fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)     # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count/fps)
        print("\n ### Frame parameter ###")
        print("Height (px): ", vidcap.get(4))
        print("Width (px): ", vidcap.get(3))
        print("frames per sec: ", fps)
        print("duration (sec): ", (frame_count/fps))
        print("Total number of images: ", int(vidcap.get(7) - 1))
        success, image_from_avi = vidcap.read()

        count = 0
        while success:
            # vidcap.set(cv2.CAP_PROP_POS_MSEC, count)    # added this line
            success, image_from_avi = vidcap.read()
            if success:
                count = count + 1
                print(count, end="\r")

                # print(len(image_from_avi), len(image_from_avi[0]), len(image_from_avi[0][0]))
                gray = cv2.cvtColor(image_from_avi, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                # make_borders_to_image
                top, bottom, left, right = [int(width / 8)] * 4
                gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                image_from_avi_temp = cv2.copyMakeBorder(image_from_avi, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                         value=0)
                avi_images_org.append(image_from_avi_temp)
                avi_images.append(gray)
                # plt.imshow(gray)
                # plt.show()
        if skip_images_factor > 1:
            return avi_images[::skip_images_factor], avi_images_org[::skip_images_factor], fps / skip_images_factor
        else:
            return avi_images, avi_images_org, fps

    def extract_images_from_image_folder(self):
        img_files = sorted(glob.glob(self.path_in + f"\\*.*"))
        imgs_temp = [cv2.imread(x) for x in img_files]
        imgs_temp_org = imgs_temp.copy()
        imgs_temp = [cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2GRAY) for x in imgs_temp]
        height, width = imgs_temp[0].shape
        print(height, width)
        if skip_images_factor > 1:
            return imgs_temp[::skip_images_factor], imgs_temp_org[::skip_images_factor]
        else:
            return imgs_temp, imgs_temp_org


def distance(point1, point2):
    # not used
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5


def fft_frequency(temp_values):
    """
    * uses rfft and rfftfreq from scipy.fft
      to extract the frequencies of sine functions after fast fourier transformation (FFT)

    :param temp_values: array-like
    :return: freq: float, frequency of highest harmonic from FFT
    """

    temp_values = temp_values / np.max(temp_values)
    num_data_points = len(temp_values)
    yf = np.abs(rfft(temp_values))
    skip_vals = 5
    xf = rfftfreq(num_data_points, 1 / frames_per_second)
    max_vals = np.where(yf[skip_vals:] > max(yf[skip_vals:]) * 0.66)[0]
    first_max_val = max_vals[0]  # skip 0° harmonic

    # plt.plot(xf, yf)
    # plt.show()

    if len(max_vals) == 1:
        freq = xf[first_max_val + skip_vals]  # skip 0° harmonic
        return round(freq, 2)
    else:
        second_max_val = np.where(yf[skip_vals:] > max(yf[skip_vals:]) * 0.66)[0][1]  # skip 0° harmonic
        if second_max_val - first_max_val < 4:
            freq = np.mean([xf[first_max_val + skip_vals], xf[second_max_val + skip_vals]])
        else:
            # freq = xf[list(yf).index(max(yf[4:]))]  # skip 0° harmonic
            freq = xf[first_max_val + skip_vals]  # skip 0° harmonic
        return round(freq, 2)


def fit_function(values):
    """
    * iteratively searches for best sine fitting function of raw data using get_best_fit()
    * done for all data points and only for one quarter of data points and compares fitting scores
      to account for instable signals
    * includes also the fft_frequency() function

    :param values: array-like
    :return: freq: float; frequency of best fitting function
    :return: fft_freq: float; frequency of FFT highest harmonic (see fft_frequency())
    :return: fitfunc_final: array-like; best fitting function
    :return: best_score: array-like; fitting score of best fitting function
    """
    # fit heart beat and extract frequencies
    t_temp1 = np.arange(0, len(values))
    best_fit = {}
    iterations = 30  # default = 10
    freq_temp = 0.4
    for i in range(iterations):
        fitfunc, freq, Amp1_t, phase1_t, fitting_score = get_best_fit(data=values, guess_freq=freq_temp,
                                                                      guess_phase=0, t_dim=t_temp1)
        f1_score = abs(np.corrcoef(fitfunc, values)[0][1])
        if f1_score > 0:
            best_fit[f1_score] = [freq, fitfunc, f1_score, phase1_t]
        else:
            best_fit[0] = [freq, fitfunc, f1_score, phase1_t]
        freq_temp = freq_temp + 0.1
        if f1_score > 0.6:
            break
    best_score = np.max(list(best_fit.keys()))

    # compare freq to truncated signal:
    best_fit_truncated = {}
    freq_temp = 0.4
    for i in range(iterations):
        val_temp = values[:int(len(values) * 0.25)]
        t_temp = t_temp1[:int(len(values) * 0.25)]
        fitfunc_temp, f_temp, Amp1_t, phase1_t, fitting_score = get_best_fit(data=val_temp, guess_freq=freq_temp,
                                                                             guess_phase=0, t_dim=t_temp)
        fit_temp_score = abs(np.corrcoef(fitfunc_temp, val_temp)[0][1])
        if fit_temp_score > 0:
            best_fit_truncated[fit_temp_score] = [f_temp, Amp1_t, phase1_t]
        else:
            best_fit_truncated[0] = [f_temp, Amp1_t, phase1_t]
        freq_temp = freq_temp + 0.1
        if fit_temp_score > 0.6:
            break

    print("Best score: ", best_score)
    if np.std(values) < 0.001:
        print("no heartbeat found")
        return False
    if best_score < 0.02:
        print("no fitting function found: ", best_score)
        best_score = best_score
        freq = best_fit[best_score][0]
        fitfunc_final = best_fit[best_score][1]
    else:
        freq, fitfunc_final, f2_score, phase1_t = \
            best_fit[best_score][0], best_fit[best_score][1], best_fit[best_score][2],  best_fit[best_score][3]
        best_score_trunc = np.max(list(best_fit_truncated.keys()))
        print("Score and Freq for complete signal: ",  round(best_score, 2), best_fit[best_score][0])
        freq_trunc = best_fit_truncated[best_score_trunc][0]
        print("Score and Freq for 1/4 signal: ", round(best_score_trunc, 2), freq_trunc)

        if abs(best_fit[best_score][0] - freq_trunc) > 0.8 and best_score < 0.6 \
                and best_score_trunc > 0.2 and best_fit[best_score][0] > freq_trunc > 0.2:
            print("Use truncated data for fitting")

            freq = freq_trunc
            A = best_fit_truncated[best_score_trunc][1]
            phase1_t = best_fit_truncated[best_score_trunc][2]  # best_fit[best_score][-1]  #
            c = np.median(values)
            w = (2 * np.pi) * (freq / frames_per_second)

            # create new sine function
            def my_sin(x, amplitude, freq_, phase, offset):
                return amplitude * np.sin(x * freq_ + phase) + offset
            fitfunc_temp2 = my_sin(t_temp1, A, w, phase1_t, c)
            fitfunc_final = fitfunc_temp2
            best_score = best_score_trunc

        # use also Fast Fourier Transformation (FFT) to extract frequency !!!
    fft_freq = fft_frequency(values)
    print("Fast Fourier Transformation (FFT) Freq: ", fft_freq)
    return round(freq, 2), fft_freq, fitfunc_final, round(best_score, 2)


def find_fish_eye(image_matrix3):
    """
    find fish eye to crop fish image
    :param image_matrix3:
    :return:
    """

    # create a mask
    x, y = np.ogrid[:len(image_matrix3), :len(image_matrix3[0])]
    mask = (x - int(len(image_matrix3) / 2)) ** 2 \
           + (y - int(len(image_matrix3[0]) / 2)) ** 2 > (len(image_matrix3) * 0.78) ** 2
    mask = 255 * mask.astype(int)

    image_matrix3[mask == 255] = np.max(image_matrix3)

    """corner = 100
    image_matrix3[:corner, :corner] = 30_000
    image_matrix3[:corner:, -corner:] = 30_000
    image_matrix3[-corner::, :corner] = 30_000
    image_matrix3[-corner:, -corner:] = 30_000"""

    eye_ = np.where(image_matrix3 < np.percentile(image_matrix3, 0.22))
    eye_ = np.asarray(eye_)
    eye_temp_ = [np.mean(eye_[0]), np.mean(eye_[1])]

    # plt.imshow(image_matrix3)
    # plt.scatter(eye_temp_[1], eye_temp_[0], s=100, c="red")
    # plt.show()

    x_start = int(eye_temp_[1]) - 260
    y_start = int(eye_temp_[0] - 260)
    y_end = int(eye_temp_[0] + 200)
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    image_matrix_temp = image_matrix3[y_start:y_end, x_start:]

    if len(image_matrix_temp) < 100:
        return

    eye_2 = np.where(image_matrix_temp < np.percentile(image_matrix_temp, 0.65))
    eye_center_1 = [int(np.mean(eye_2[0])), int(np.percentile(eye_2[1], 35))]  # y, x

    """plt.imshow(image_matrix_temp)
    plt.scatter(eye_center_1[1], eye_center_1[0], s=100, c="red")
    plt.show()"""

    # image_matrix_fish = np.copy(image_matrix_temp)
    # fish = np.where(image_matrix_temp < np.percentile(image_matrix_temp, 4.4))
    # image_matrix_fish[fish] = 30_000

    return eye_center_1, eye_temp_


def get_std_dict(x_start_temp, x_end_temp, y_start_temp, y_end_temp, cut_off=5):
    std_matrix = []
    std_max_dict = {}
    n_points = 0
    found_150 = False
    for y in range(y_start_temp, y_end_temp):
        std_max_array = []
        if found_150:
            break
        for x in range(x_start_temp, x_end_temp):
            std_value = np.std(img_matrices[:, y, x])
            std_max_array.append(std_value)
            if std_value > cut_off:  # \
                # and distance([y, x], eye_center) < 180 \
                # and (y >= eye_center[0] and x >= eye_center[1]):  # x,y right, close to eye !!!!
                n_points += 1
                std_max_dict[(x, y)] = img_matrices[:, y, x]
                if n_points == 150:
                    found_150 = True
                    break
        std_matrix.append(std_max_array)
    return std_matrix, std_max_dict


def get_heartbeat():
    n_points = 0
    if os.path.isfile(store_in_temp + fr"\{fish_name}.npy") and not overwrite_raw_values:
        values = np.load(store_in_temp + fr"\{fish_name}.npy")
        std_matrix = np.load(store_in_temp + fr"\{fish_name}_std_matrix.npy")
        xs = np.load(store_in_temp + fr"\{fish_name}_xs.npy")
        ys = np.load(store_in_temp + fr"\{fish_name}_ys.npy")
    else:

        # img_matrices = img_matrices[5:65]
        # quit()
        n_points = 0

        x_dim, y_dim = len(img_matrices[0][0]), len(img_matrices[0])
        x_start_temp = int(heart_center[1]) - 40
        x_end_temp = int(heart_center[1]) + 80
        y_start_temp = int(heart_center[0] - 40)
        y_end_temp = int(heart_center[0] + 80)
        if x_start_temp < 0:
            x_start_temp = 0
        if x_end_temp > x_dim:
            x_end_temp = x_dim - 1
        if y_start_temp < 0:
            y_start_temp = 0
        if y_end_temp > y_dim:
            y_end_temp = y_dim - 1

        for cut in [24, 20, 18, 16, 14, 12, 10, 8, 6, 4]:
            std_matrix, std_max_dict = get_std_dict(x_start_temp, x_end_temp, y_start_temp, y_end_temp, cut_off=cut)
            if len(std_max_dict) != 0:
                break
        if len(std_max_dict) == 0:
            print("not found")
            return False

        sorted_px = sorted(std_max_dict, key=lambda x1: np.mean(x1))
        values, xy = [], []
        for idx, elem in enumerate(sorted_px):
            values.append(std_max_dict[elem])
            xy.append(elem)
            if idx == 10:
                break
        xs, ys = [], []
        for px in xy:
            xs.append(px[0])
            ys.append(px[1])

        values = np.asarray(values)
        values = np.mean(values, axis=0)
        values = stats.zscore(values, ddof=1)
        values = np.array(values)

        np.save(store_in_temp + fr"\{fish_name}.npy", values)
        np.save(store_in_temp + fr"\{fish_name}_std_matrix.npy", std_matrix)
        np.save(store_in_temp + fr"\{fish_name}_xs.npy", xs)
        np.save(store_in_temp + fr"\{fish_name}_ys.npy", ys)

    max_diff_ = round(np.max(values) - np.min(values), 2)
    out_temp = fit_function(values)
    if not out_temp:
        return False
    freq, fft_freq, fitfunc, fit_score = out_temp
    return freq, fft_freq, fit_score, n_points, xs, ys, std_matrix, values, fitfunc, max_diff_


def plot_heart_beat(f1_, f2_fft, fit_score, n_points, xs, ys, std_matrix, values, fitfunc):

    if len(xs) == 0:
        return

    # heart
    plt.figure(n_points * 10, figsize=(6, 4))
    # plt.title(fish_name)
    plt.subplots_adjust(left=0.135, wspace=0.23, hspace=0.4, bottom=0.14, top=0.9, right=0.9)

    plt.subplot2grid((2, 4), (0, 0), colspan=3, rowspan=1)
    # increase contrase
    image_show_temp = cv2.LUT(image_show, lookUpTable)

    dark_orange = [236, 117, 7]
    light_orange = [250, 220, 180]
    image_show_temp[std_matrix_temp > np.max(std_matrix_temp) * 0.30] = light_orange
    image_show_temp[std_matrix_temp > np.max(std_matrix_temp) * 0.50] = dark_orange

    # plot movement
    # plt.imshow(std_matrix_temp, cmap="Oranges", alpha=0.3)
    # plt.scatter(x=eye_center[1], y=eye_center[0], c="blue", s=20)
    plt.imshow(image_show_temp)
    plt.axis("off")
    plt.title("Zebrafish", c='black')

    rect = Rectangle(xy=(int(heart_center[1]) - 60, int(heart_center[0]) - 60),
                     width=120, height=120, fill=False,
                     color="white", linewidth=0.7, linestyle="dashed")
    plt.gca().add_patch(rect)

    plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)
    plt.title("Heart", c="black")

    img_temp = image_show_temp[
               int(heart_center[0]) - 80: int(heart_center[0]) + 80,
               int(heart_center[1]) - 80: int(heart_center[1]) + 80]
    if len(img_temp) == 0:
        return

    try:
        plt.imshow(img_temp)
    except ValueError:
        pass

    plt.axis("off")
    plt.subplot2grid((2, 4), (1, 0), rowspan=1, colspan=4)
    x_values = [i_/len(values) * cut_movie_at for i_ in range(1, len(values) + 1)]
    plt.plot(x_values, values, color="black", label="raw signal",  linewidth=1)

    if f1_ == 0:
        plt.plot(x_values, [0] * len(values), linestyle="dashed", color='#EC7507',
                 label="best fit", linewidth=1.4,)
    else:
        plt.plot(x_values, fitfunc, linestyle="dashed", color='#EC7507',
                 label="best fit", linewidth=1.4,)
    plt.hlines(xmin=0, xmax=max(x_values), y=np.mean(values), linestyles="--", colors="gray")
    plt.text(x=max(x_values) * 1.1, y=np.mean(values), s=fish_name, rotation=270, va="center", ha="center")

    plt.title("freq=%s s$^{-1}$, fft_freq=%s s$^{-1}$ (fitting score=%s)" % (f1_, f2_fft, fit_score),
              fontdict={"size": 10.5})
    plt.legend(fontsize=6, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1)
    plt.ylabel("Heartbeat (RU)")
    plt.xlabel(f"Time (s) / {frames_per_second} frames/s")
    plt.savefig(rf"{store_in}\{fish_name}_heart_beat.png")
    plt.savefig(rf"{store_in}\{fish_name}_heart_beat.pdf")
    # plt.show()
    plt.close()
    pass


def get_colormap_for_values(values):
    """
    :param values:
    :return colormap:
    """
    beta_colors = []
    rgb = list(cm.get_cmap(name='Oranges')(np.arange(0, 256)))
    max_beta = max(values)
    min_beta = min(values)
    if abs(min_beta) > abs(max_beta):
        max_beta = abs(min_beta)
    for beta in values:
        temp_beta = beta / max_beta
        index = 127 + int(128 * temp_beta)
        if index == -1:
            index = 0
        beta_colors.append(rgb[index])
    return beta_colors, max_beta


def get_best_fit(data, guess_freq, guess_phase, t_dim):
    """
    * fits raw data to a sine function using scipy.optimize.curve_fit
    * returns params to describe fitting sine function

    :param data: array-like
    :param guess_freq: float
    :param guess_phase: float
    :param t_dim: array-like
    :return: fitfunc: array-like
    :return: A:  amplitude of sine function, float
    :return: p:  phase of sine function, float
    :return: fit_score:  fitting score of fitting function

    """
    guess_amplitude = 3 * np.std(data)
    guess_offset = np.median(data)
    p0 = [guess_amplitude, guess_freq, guess_phase, guess_offset]

    # create the function we want to fit
    def my_sin(x, amplitude, freq_, phase, offset):
        return amplitude * np.sin(x * freq_ + phase) + offset

    popt, pcov = scipy.optimize.curve_fit(my_sin, t_dim, data, p0=p0, maxfev=10000)  # default 10000
    A, w, p, c = popt
    f = w / (2. * np.pi)
    A, p, f = round(3 * np.std(data), 1), round(p, 1), round(f * frames_per_second, 2)  # frame/s
    if c / guess_offset > 2 or guess_offset / c > 2 or c < 0:
        c = np.mean(data) * 1.4
    fitfunc = my_sin(t_dim, A, w, p, c)
    fit_score = np.corrcoef(fitfunc, data)[0][1]
    if fit_score < 0:
        fitfunc = my_sin(t_dim, -A, w, p, c)

    return fitfunc, f, abs(A), p, abs(fit_score)


def fig2img(fig_):
    buf = io.BytesIO()
    fig_.savefig(buf)
    buf.seek(0)
    return buf


def create_gifs_from_images():
    images_for_gif = []
    images_for_gif2 = []
    # prepare for GIF
    time_ = 0
    for idx, img in enumerate(img_matrices):
        matrix_temp = img[
                      int(heart_center[0]) - 80: int(heart_center[0]) + 80,
                      int(heart_center[1]) - 80: int(heart_center[1]) + 80]
        fig = plt.figure(idx)
        plt.imshow(matrix_temp, cmap="coolwarm")
        steps_ = cut_movie_at / len(images_final)
        time_ = time_ + steps_
        plt.title(f"{fish_name}:\n{idx + 1}/{len(images_final)} {round(time_, 2)} s")
        plt.axis("off")
        images_for_gif.append(fig2img(fig))
        plt.close("all")

        fig = plt.figure(idx + 1000)
        plt.imshow(img, cmap="coolwarm")
        plt.title(f"{fish_name}:\n{idx + 1}/{len(images_final)} {round(time_, 2)} s")
        plt.axis("off")
        images_for_gif2.append(fig2img(fig))
        plt.close("all")

    # create gifs and store in project subfolder
    # fp_in = sub_output_path + "/*temp.png"
    if len(images_for_gif) != 0:
        # img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime)]
        img, *imgs = [Image.open(f) for f in images_for_gif]
        img.save(fp=store_in + rf"\{fish_name}_GIF_HEART.gif",
                 format='GIF', append_images=imgs,
                 save_all=True,
                 duration=int((1000/frames_per_second) / 1),  # 1/2 = 2x speed of GIFs; 1 = original
                 # The display duration of each frame, in milliseconds.
                 loop=10)
        # for file in glob.glob(fp_in):
        #     os.remove(file)
        for stream in images_for_gif:
            stream.close()  # otherwise memory overload

    if len(images_for_gif2) != 0:
        # img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime)]
        img, *imgs = [Image.open(f) for f in images_for_gif2]
        img.save(fp=store_in + rf"\{fish_name}_GIF_FISH.gif",
                 format='GIF', append_images=imgs,
                 save_all=True,
                 duration=int((1000/frames_per_second) / 1),  # 1/2 = 2x speed of GIFs; 1 = original
                 # The display duration of each frame, in milliseconds.
                 loop=10)
        # for file in glob.glob(fp_in):
        #     os.remove(file)
        for stream in images_for_gif2:
            stream.close()  # otherwise memory overload


def parse_arguments():
    """
    * takes arguments defined in heart_beat_GUI.py and heart_beat_GUI_MAIN.py

    :return: args_
    """
    parser = argparse.ArgumentParser(usage="define input path and file name for .csi file",
                                     description="python arg_paser -s 120575 SUM_STAT ./",)
    parser.add_argument('input', type=str, help='path to .csi file'),
    parser.add_argument('-o', '--output', type=str, help='directory of output result file, default:working directory',
                        default='./'),
    parser.add_argument('--overwrite', action="store_true", help='overwrite flag',
                        default=False),
    parser.add_argument('--skip_images', type=int, help='Skip every xx image for faster analysis',
                        default=1),
    parser.add_argument('-n', '--name', type=str, help='define project name', default="Heart_beat", metavar=''),
    parser.add_argument('-t', '--threads', type=int, help='number of threads',
                        default=1, metavar=''),
    parser.add_argument('-f', '--resize_factor', type=float, help='Resize the image for faster processing [0.1 - 1.0]',
                        default=1.0, metavar=''),
    parser.add_argument('-fs', '--frames_per_sec', type=float, help='Number of frames per second',
                        default=10, metavar=''),
    parser.add_argument('-ps', '--pixel_size', type=float, help='length of on pixel (e.g. 1.5 µm / pixel)',
                        default=1.5, metavar=''),
    parser.add_argument('-cut', '--cut_movie_at', type=float, help='e.g. cut movie at 20 s to normalize movie length',
                        default=20.0, metavar=''),
    parser.add_argument('-img_counter', '--image_counter', type=str, help='e.g. Number of image analyzed',
                        default="1/100", metavar=''),
    parser.add_argument('-ff', '--file_format', type=str, help='e.g. .tif or .avi',
                        default=".avi", metavar=''),
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':

    # load params
    args = parse_arguments()
    resize_factor, frames_per_second = args.resize_factor, args.frames_per_sec
    skip_images_factor = int(args.skip_images) + 1  # default 0 + 1 = 1 for array[::1] > takes every image !!!!
    pixel_size, cut_movie_at, file_format = args.pixel_size, args.cut_movie_at, args.file_format  # µm per pixel
    image_counter = args.image_counter
    scale_bar_size = 50 / pixel_size  # 50 µm scale bar
    overwrite_raw_values = bool(args.overwrite)
    main_project, movie_path,  main_output_path = args.name, args.input, args.output
    condition_folder_name = fish_heart_file = movie_path.split("\\")[-1]
    condition_folder_name = condition_folder_name.replace(f"{file_format}", "").replace(",", "_").replace(" ", "_")
    sub_output_path = main_output_path + "\\" + condition_folder_name
    if not os.path.isdir(sub_output_path.replace("/", "\\")):
        os.mkdir(sub_output_path.replace("/", "\\"))

    print("start pyHeart4Fish bright field module")

    # extract images from folders/ videos
    images, org_images, image_container = [], [], []
    image_extracter = ExtractImages(movie_path)
    if file_format == ".avi":
        images, org_images, frames_per_second = image_extracter.extract_images_from_avi()
    elif file_format == ".czi":
        images,  org_images,frames_per_second = image_extracter.extract_images_from_czi()
    elif file_format == ".mp4":
        images, org_images, frames_per_second = image_extracter.extract_images_from_mp4()
    elif file_format in [".tif", ".jpeg", ".png"]:
        images, org_images = image_extracter.extract_images_from_image_folder()
        frames_per_second = frames_per_second
    images_final = images[:int(cut_movie_at * frames_per_second)]
    max_value = np.max(images_final)
    # images_final = [x/max_value)
    images_final = [cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for x in images_final]
    images_final_org = org_images[:int(cut_movie_at * frames_per_second)]

    store_in = main_output_path
    store_in_temp = rf"{main_output_path}\{condition_folder_name}"
    if not os.path.isdir(store_in_temp):
        os.mkdir(store_in_temp)

    # prepare look-up table to increase contrast
    lookUpTable = np.empty((1, 256), np.uint8)
    gamma = 0.6
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    output = dict()
    fish_name = condition_folder_name
    output["Project"] = main_project
    output["Condition"] = fish_name
    print("\n", fish_name)
    image_matrices = []

    first_img = True
    eye_temp = [0, 0]
    image_show = images_final_org[0]
    for idx, image_matrix in enumerate(images_final):

        print(idx, end="\r")
        # print(np.max(image_matrix))
        # print(np.mean(image_show))
        # image = img
        # image_show = cv2.imread(img)
        # print(image_matrix.shape, image_show.shape)
        y_min, y_max = int(0.31 * len(image_matrix)), int(0.64 * len(image_matrix))
        x_min, x_max = int(0.12 * len(image_matrix[0])), int(0.82 * len(image_matrix[0]))
        image_matrix2 = np.asarray(image_matrix[y_min:y_max, x_min:x_max])
        if idx == 0:
            image_show = image_show[y_min:y_max, x_min:x_max]

        if first_img:
            eye_center, eye_temp = find_fish_eye(image_matrix2.copy())
            # ToDo: Flip fish if eyes is right !!!
            first_img = False

        x_dim, y_dim = len(image_matrix2[0]), len(image_matrix2)
        x_start = int(eye_temp[1]) - 260
        x_end = x_dim
        y_start = int(eye_temp[0] - 260)
        y_end = int(eye_temp[0] + 200)
        if x_start < 0:
            x_start = 0
        if x_end > x_dim:
            x_end = x_dim - 1
        if y_start < 0:
            y_start = 0
        if y_end > y_dim:
            y_end = y_dim - 1

        image_matrix_new = image_matrix2[y_start: y_end, x_start: x_end]
        if idx == 0:
            image_show = image_show[y_start: y_end, x_start: x_end]
        image_matrices.append(image_matrix_new)
    img_matrices = np.asarray(image_matrices)

    # create StdDev matrix to find area with most movement
    std_matrix_temp = img_matrices.std(0)

    print(np.mean(std_matrix_temp))
    # quit()
    std_matrix_temp[std_matrix_temp < 3] = 0
    # plt.imshow(image_show)
    # plt.imshow(std_matrix_temp, cmap="Oranges", alpha=0.6)
    # plt.show()
    heart_area = np.where(
        std_matrix_temp[:, :int(len(std_matrix_temp[0]) / 2.6)] >
        np.max(std_matrix_temp[:, :int(len(std_matrix_temp[0]) / 2.6)]) * 0.75)

    if len(heart_area[0]) == 0:
        print("NO HEART AREA / HEARTBEAT FOUND!!")
        output["Freq"] = np.nan
        output["Freq_FFT"] = np.nan
        output["Fit_score"] = np.nan
        # output["Rel_contraction"] = np.nan
        df = pd.DataFrame(data=output, index=[0])
        df.to_csv(rf"{store_in_temp}\{fish_name}.csv", index=False)
        quit()

    heart_center = np.percentile(heart_area, 5, axis=1)

    # find black eyes to cut images
    eye = np.where(img_matrices[0] < np.min(img_matrices[0] * 1.2))
    eye_center_temp = np.mean(eye, axis=1)  # y, x
    new_center = np.mean(list(zip(heart_center, eye_center_temp)), axis=1)

    # create whole body and heart area GIF to control heartbeat manually
    create_gifs_from_images()

    max_dev = np.max(std_matrix_temp)
    print(max_dev)
    if max_dev < 3.6 or (max_dev < 5.8 and abs(heart_center[1] - eye_center_temp[1]) > 150):
        print("NO HEARTBEAT FOUND!!")
        output["Freq"] = np.nan
        output["Freq_FFT"] = np.nan
        output["Fit_score"] = np.nan
        # output["Rel_contraction"] = np.nan
        df = pd.DataFrame(data=output, index=[0])
        df.to_csv(rf"{store_in_temp}\{fish_name}.csv", index=False)
        quit()

    # parameter not used yet > too imprecise !
    area_vessels = len(np.where(std_matrix_temp > np.max(std_matrix_temp) * 0.15)[0])

    steps = np.arange(0, len(image_matrices))
    steps = [float(x) for x in steps]
    t = np.asarray(steps)

    # get heartbeat by using 10 pixel with highest StdDev in heart area
    out = get_heartbeat()
    # out: freq, fft_freq, fit_score, n_points, xs, ys, std_matrix, values, max_diff_, area_vessel
    if not out:
        output["Freq"] = 0
        output["Freq_FFT"] = 0
        output["Fit_score"] = 0
        # output["Rel_contraction"] = 0
        df = pd.DataFrame(data=output, index=[0])
        df.to_csv(rf"{store_in_temp}\{fish_name}.csv", index=False)
        quit()

    # plot heartbeat as sine curve
    plot_heart_beat(*out[:-1])

    # store data
    output["Freq"] = out[0]
    output["Freq_FFT"] = out[1]
    output["Fit_score"] = out[2]
    print("\n")
    df = pd.DataFrame(data=output, index=[0])
    df.to_csv(rf"{store_in_temp}\{fish_name}.csv", index=False)
    plt.close("all")

