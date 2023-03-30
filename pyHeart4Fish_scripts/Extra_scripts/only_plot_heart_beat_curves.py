import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy import stats
from scipy.fft import rfft, rfftfreq
from tkinter import filedialog
import os
import matplotlib.font_manager as font_manager


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


def fft_frequency(temp_values):
    """
    * uses rfft and rfftfreq from scipy.fft
      to extract the frequencies of sine functions after fast fourier transformation (FFT)

    :param temp_values: array-like
    :return: freq: float, frequency of highest harmonic from FFT
    """

    temp_values = temp_values / np.max(temp_values)
    num_data_points = int(frames_per_second * cut_movie_at)
    num_data_points = len(ventricle_values)
    yf = np.abs(rfft(temp_values))
    xf = rfftfreq(num_data_points, 1 / frames_per_second)
    print(yf, xf)
    # plt.plot(xf, yf)
    # plt.show()
    freq = xf[list(yf).index(max(yf[4:]))]  # skip 0° harmonic
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
    freq_temp = 0.2
    for i in range(iterations):
        fitfunc, freq, Amp1_t, phase1_t, fitting_score = get_best_fit(data=values, guess_freq=freq_temp,
                                                                      guess_phase=0, t_dim=t_temp1)
        f1_score = abs(np.corrcoef(fitfunc, values)[0][1])
        if f1_score > 0:
            best_fit[f1_score] = [freq, fitfunc, f1_score, phase1_t]
        else:
            best_fit[0] = [freq, fitfunc, f1_score, phase1_t]
        freq_temp = freq_temp + 0.05
        if f1_score > 0.18:
            break
    best_score = np.max(list(best_fit.keys()))

    # compare freq to truncated signal:
    best_fit_truncated = {}
    freq_temp = 0.2
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
        freq_temp = freq_temp + 0.05
        if fit_temp_score > 0.22:
            break

    print("Best score: ", best_score)
    if np.std(values) < 0.01:
        print("no heartbeat found")
        return 0, 0, [], 0
    if best_score < 0.02:
        print("no fitting function found: ", best_score)
        best_score = 0
        return 0, 0, [], best_score
    else:
        freq, fitfunc_final, f2_score, phase1_t = \
            best_fit[best_score][0], best_fit[best_score][1], best_fit[best_score][2],  best_fit[best_score][3]
        best_score_trunc = np.max(list(best_fit_truncated.keys()))
        print("Score and Freq for complete signal: ",  round(best_score, 2), best_fit[best_score][0])
        freq_trunc = best_fit_truncated[best_score_trunc][0]
        print("Score and Freq for 1/4 signal: ", round(best_score_trunc, 2), freq_trunc)

        if best_score_trunc >  0.8 or (abs(best_fit[best_score][0] - freq_trunc) > 0.8 and best_score < 0.6 \
                and best_score_trunc > 0.2 and best_fit[best_score][0] > freq_trunc > 0.2):
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


def calc_phase_shift_by_cross_correl(val1, val2, steps=0.05):
    """
    * creates cross-correlation function using by shifting atrium z-scored raw data against ventricle z-scored raw data

    :param val1: array-like
    :param val2: array-like
    :param steps: float; [0 - 0.2]
    :return: phase_shift: float; first maximum of cross-correlation function
    :return: correls_temp: cross-correlation function
    """
    # take middle of signal
    t1 = int(steps * len(val1))
    t2 = int(steps + 0.4 * len(val1))
    correls_temp = dict()
    for i in range(0, int(len(val1) / 6) - 1):
        a = np.corrcoef(val1[t1 + i: t2 + i], val2[t1:t2])[0][1]
        i = i/len(val1) * cut_movie_at
        correls_temp[i] = a
    corr_vals = np.asarray(list(correls_temp.values()))
    maxima = np.where(corr_vals > np.percentile(corr_vals, 65))
    time_dim = list(correls_temp.keys())
    phase_shift = round(time_dim[maxima[0][0]], 2)
    return phase_shift, correls_temp


def include_area_values_func(val, area_val):
    """
    * includes he inverse of area values in intensity values to account for contraction

    :param val: array-like
    :param area_val: array-like
    :return: val_new: array-like
    """
    area_val[area_val == 0] = np.median(area_val)
    area_values_temp = [1 / x for x in area_val]
    area_values_temp = [x / np.max(area_values_temp) for x in area_values_temp]
    val_new = [((x + y)/2) ** 2 for x, y in zip(val, area_values_temp)]
    return np.asarray(val_new)


def smoothen_data(raw_data):
    """
    * makes raw data smoother to reduce outliers

    :param raw_data: array-like
    :return: data_temp: array-like
    """
    data_temp = []
    steps = 2
    for i in range(0, len(raw_data) - steps):
        data_temp.append(np.mean(raw_data[i:i + steps]))
    return np.asarray(data_temp)


def estimated_autocorrelation(x):
    """
    estimates the auto-correlation of a function to obtain a score for regularity

    :param x: signal values as array
    :return: estimate: float
    """
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    result = sorted(result, reverse=True)
    # print(result)
    a = float(np.mean(result[1:5]))
    estimate = round(a, 2)
    return estimate


def process_raw_data_and_plot_heart_beat_curve():
    """
    * process raw data and extracts several parameter
    to characterize atrium and ventricle contraction and overall heart beat
    * calculates and extracts:
     > auto-correlation cooefication to obtain a score for regularity
     > frequencies and fitting function for heart beat data using fit_function()
     > phase shift between atrium raw signal in ventricle raw signal
     > arrhythmia_score
              >> arrhythmia_score > 0.7 means can mean that
                1) the delay/ phase is very low
                2) the atrium contraction differ significantly from ventricle contraction
                3) atrium or ventricle show arrhythmia
                4) 1. degree av-block might be present
                The lower arrhythmia_score, the more regular the heart beat and vice versa
     > av_block_score
              >> take absolut difference from frequencies
              >>av_block_score > 0.5 means that an av-block is detected > 2. / 3. degree av-block
     > stores output values in value_dict
     > plots heart beat curves for atrium and ventricle and merge
    """

    auto_corr_coeff = estimated_autocorrelation(atrium_values)
    auto_corr_coeff2 = estimated_autocorrelation(ventricle_values)

    # fit raw heartbeat data
    print("\nfit heartbeat and extract frequencies for atrium")
    f1_atrium, f1_fft_atr, fitfunc_atrium, fit_score_all_atr_temp = fit_function(atrium_values)
    print("\nfit heartbeat and extract frequencies for ventricle")
    f1_ventricle, f1_fft_ventr, fitfunc_ventricle, fit_score_all_ventr_temp = fit_function(ventricle_values)

    # calculate phase shift again
    atrium_values_norm = stats.zscore(atrium_values, ddof=1)
    ventricle_values_norm = stats.zscore(ventricle_values, ddof=1)
    phase_shift, corr_func = calc_phase_shift_by_cross_correl(atrium_values_norm, ventricle_values_norm)

    # calculate sum of difference function to access arrhythmia
    diff_val = [abs(x+y) for x, y in zip(atrium_values_norm, ventricle_values_norm)]
    arrhythmia_score = round(np.sum(diff_val) / len(diff_val), 2)
    conduction_score = round(abs(f1_atrium - f1_ventricle) + abs(f1_fft_atr - f1_fft_ventr), 2)

    ############################################################################################################
    # put values into value dictionary ###
    ############################################################################################################
    # area
    max_atrium = np.median(sorted(atrium_area_values)[-3:])
    min_atrium = np.median(sorted(atrium_area_values)[:4])
    max_ventricle = np.median(sorted(ventricel_area_values)[-3:])
    min_ventricle = np.median(sorted(ventricel_area_values)[:3])

    # ToDo: Calculate Vol differently
    # V_cylinder = area * height > approximation !
    max_vol_atrium = max_atrium * max_atrium**0.5
    min_vol_atrium = min_atrium * min_atrium**0.5
    max_vol_ventr = max_ventricle * max_ventricle**0.5
    min_vol_ventr = min_ventricle * min_ventricle**0.5

    font = font_manager.FontProperties(family="Times New Roman",
                                       weight='regular',
                                       style='normal', size=7.5)

    # plot heartbeat signals
    plt.figure(fish_heart_file, figsize=(7, 6))
    plt.title(fish_heart_file)
    plt.subplots_adjust(hspace=0.6)
    x_values = [i/len(ventricle_values) * cut_movie_at for i in range(1, len(ventricle_values) + 1)]

    plt.subplot2grid((3, 1), (0, 0))  # position 1 !!!!!!!!!!!
    plt.hlines(xmin=0, xmax=max(x_values), y=np.mean(atrium_values), linestyles="--", colors="gray")
    plt.plot(x_values, atrium_values, label="raw signal", c="black", linewidth=1.5)
    plt.title("Atrium: freq=%s s$^{-1}$, fft_freq=%s s$^{-1}$" % (f1_atrium, f1_fft_atr),
              fontdict={"size": 10, "font": "Times New Roman"})
    if len(fitfunc_atrium) != 0:
        plt.plot(x_values, fitfunc_atrium, linestyle="dashed", color="red", label="best fit", linewidth=1)

    if len(fitfunc_atrium) != 0:
        plot_y_min2, plt_y_max2 = min([min(fitfunc_atrium), min(atrium_values)]) - .2, \
                                  max([max(fitfunc_atrium), max(atrium_values)]) + .2
    else:
        plot_y_min2, plt_y_max2 = min(atrium_values) - .2,  max(atrium_values) + .2

    if plot_y_min2 > 0.4:
        plot_y_min2 = 0.2
    plt.ylim(plot_y_min2, plt_y_max2)
    plt.legend(fontsize=7.5, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1, prop=font,)
    plt.ylabel("Heartbeat (RU)", va="bottom", ha="center", fontdict={"font": "Times New Roman"})
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.subplot2grid((3, 1), (1, 0))  # position 2 !!!!!!!!!!!
    plt.hlines(xmin=0, xmax=max(x_values), y=np.mean(ventricle_values), linestyles="--", colors="gray")
    plt.title("Ventricle: freq=%s s$^{-1}$, fft_freq=%s s$^{-1}$" % (f1_ventricle, f1_fft_ventr),
              fontdict={"size": 10, "font": "Times New Roman"})
    plt.plot(x_values, ventricle_values, label="raw signal", c="black", linewidth=1.5)
    if len(fitfunc_ventricle) != 0:
        plt.plot(x_values, fitfunc_ventricle, linestyle="dashed", color="red", label="best fit", linewidth=1)

    plt.legend(fontsize=7.5, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1, prop=font,)
    plt.ylabel("Heartbeat (RU)", va="bottom", ha="center", fontdict={"font": "Times New Roman"})

    plt.text(x=max(x_values) * 1.1, y=np.mean(atrium_values), rotation=270,
             s=fish_heart_file, fontdict={"size": 8, "font": "Times New Roman"}, ha="center", va="center")

    if len(fitfunc_ventricle) != 0:
        plot_y_min, plt_y_max = min([min(fitfunc_ventricle), min(ventricle_values)]) - .2, \
                                max([max(fitfunc_ventricle), max(ventricle_values)]) + .2
    else:
        plot_y_min, plt_y_max = min(ventricle_values) - .2, max(ventricle_values) + .2

    if plot_y_min > 0.4:
        plot_y_min = 0.2
    plt.ylim(plot_y_min, plt_y_max)
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")

    plt.subplot2grid((3, 1), (2, 0))  # position 3 !!!!!!!!!!!
    title_text = f"Atrium vs. Ventricle\n(phase shift: {phase_shift} s, " \
                 f"arrhythmia score:  {arrhythmia_score}, conduction score: {conduction_score})"
    plt.title(title_text, fontdict={"size": 10, "font": "Times New Roman"})
    plt.plot(x_values, atrium_values_norm, label="Atrium", c="midnightblue", linewidth=1, linestyle="solid")
    plt.plot(x_values, ventricle_values_norm, label="Ventricle", c="darkorange", linewidth=1, linestyle="solid")
    plt.legend(fontsize=7.5, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1, prop=font,)
    plt.ylim(min(min(atrium_values_norm), min(ventricle_values_norm)) - .2,
             max(max(atrium_values_norm), max(ventricle_values_norm)) + 0.4)
    plt.ylabel("Heartbeat (RU)", fontdict={"font": "Times New Roman"})
    plt.xlabel(f"Time (s) / {frames_per_second} frames/s", fontdict={"font": "Times New Roman"})
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")
    plt.savefig(main_output_path + "/%s_frequencies_fps%s.png" % (folder, frames_per_second))
    plt.savefig(main_output_path + "/%s_frequencies_fps%s.pdf" % (folder, frames_per_second))
    plt.close()


def include_area_values_func(val, area_val):
    """
    * includes he inverse of area values in intensity values to account for contraction

    :param val: array-like
    :param area_val: array-like
    :return: val_new: array-like
    """
    area_val[area_val == 0] = np.median(area_val)
    area_values_temp = [1 / x for x in area_val]
    area_values_temp = [x / np.max(area_values_temp) for x in area_values_temp]
    val_new = [((x + y)/2) ** 2 for x, y in zip(val, area_values_temp)]
    return np.asarray(val_new)


if __name__ == '__main__':

    include_area_values = True

    main_output_path = filedialog.askdirectory()
    name = main_output_path.split("/")[-1]
    for folder in os.listdir(main_output_path):
        if not os.path.isdir(main_output_path + "/" + folder):
            continue
        # main_output_path = main_output_path + "/" + folder
        print(main_output_path)
        fish_heart_file = name
        atrium_values = np.load(main_output_path + rf"\{folder}\atrium.npy")
        ventricle_values = np.load(main_output_path + rf"\{folder}\ventricel.npy")
        atrium_area_values = np.load(main_output_path + rf"\{folder}\atrium_area.npy")
        ventricel_area_values = np.load(main_output_path + rf"\{folder}\ventricel_area.npy")

        if include_area_values:
            atrium_values = include_area_values_func(atrium_values, atrium_area_values)
            ventricle_values = include_area_values_func(ventricle_values, ventricel_area_values)

        print(len(atrium_values), len(ventricle_values))
        cut_movie_at = 6
        frames_per_second = 6
        atrium_values = atrium_values[:cut_movie_at * frames_per_second]
        ventricle_values = ventricle_values[:cut_movie_at * frames_per_second]

        process_raw_data_and_plot_heart_beat_curve()
