"""
Author: Dr. Tobias Reinberger
Data: 07.05.2022
Version: beta 1.2 for avi or czi movies
"""

import sys
import json
from PIL import Image, ImageEnhance, ImageTk
import PIL
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pandas as pd
from scipy.ndimage import rotate
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.fft import rfft, rfftfreq
from aicsimageio.readers import CziReader
import czifile
from threading import Thread
import multiprocessing as mp
from tkinter import *
from tkinter import messagebox
from tkinter import ttk, ALL
import os
import argparse
import cv2
import timeit
import io


class MainWindow:
    """
    * The main window/ application includes the rotation method and define atrium and ventricle method
    * Images are adjusted to the display geometry
    * write data into "config_file.json"-file which is later automatically moved to the pjoject subfolder

    * process_other_images_and_plot_hearts() function is executed in multiple threads using p.starmap method
    * the number of available processors is defined by the operating system
    > see https://superfastpython.com/multiprocessing-pool-starmap/
    """

    def __init__(self, root, img, all_images):
        super().__init__()

        self.root = root
        self.images = all_images
        self.logo_image = PIL.Image.open("Logo/PyHeart4Fish_PNG.png")
        self.img = img
        self.image = PIL.Image.fromarray(self.img)
        self.w, self.h = self.image.size
        display_width, display_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        display_width, display_height = int(display_width * 0.6), int(display_height * 0.7)
        self.resize_factor = self.h / display_height
        # print(self.resize_factor)
        if self.resize_factor > 1:
            self.image = self.image.resize((int(self.w/self.resize_factor), int(self.h/self.resize_factor)))
        if self.h < 200:
            self.image = self.image.resize((int(self.w * 3), int(self.h * 3)))
        self.root.geometry(f'{int(display_width)}x{int(display_height)}+50+50')
        self.root.title("pyHeart4Fish: " + image_counter + " analyzed")
        self.root.iconbitmap("Logo/PyHeart4Fish.ico")
        self.root.resizable(True, True)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # rotation window
        self.lb = Label(self.root, text="1) Rotate Image", font=("Bahnschrift", 16))
        self.lb.grid(row=0, column=0,  pady=2, sticky=S)
        self.s = ttk.Scale(self.root, from_=0, to=360, orient=HORIZONTAL, length=300,
                           command=self.rotate_image)
        self.show_rotate_info = True
        self.s.grid(row=0, column=0, sticky=E)
        self.b1 = ttk.Button(self.root, text="OK", command=self.define_areas)
        self.b1.grid(row=0, column=1, ipady=10, ipadx=10, pady=20, sticky=N)
        self.play_movie_btn = ttk.Button(self.root, text="Play Movie", command=lambda: os.startfile(movie_path))
        self.play_movie_btn.grid(row=0, column=2, ipady=10, ipadx=10, pady=20, sticky=N)

        self.next_fish = ttk.Button(self.root, text="Next Fish", command=self.go_to_next_fish)
        self.next_fish.grid(row=0, column=3, ipady=10, ipadx=10, pady=20, sticky=N)
        self.exit_fish = ttk.Button(self.root, text="Exit", command=self.exit_processing)
        self.exit_fish.grid(row=0, column=4, ipady=10, ipadx=10, pady=20, sticky=N)
        # self.restart = ttk.Button(self.root, text="Restart Fish", command=self.restart_fish)

        # define area
        self.lb2 = Label(self.root, text="2) Define Atrium and Ventricle",
                         font=("Bahnschrift", 16))

        self.b2 = ttk.Button(self.root, text="Reset", command=self.reset_areas)
        self.b3 = ttk.Button(self.root, text="OK", command=self.start_progressing)
        self.b4 = ttk.Button(self.root, text="Help", command=self.need_help)

        self.lb3 = Label(self.root, text="3) Processing images", font=("Bahnschrift", 16))
        self.progress1 = ttk.Progressbar(root, orient=HORIZONTAL, length=300, mode='determinate')
        # self.progress1_explain = Label(self.root, text="0%", font=("Bahnschrift", 7))

        self.canvas1 = Canvas(self.root, cursor="heart",  height=display_height, width=display_width, background="black")
        self.canvas1.grid(row=1, column=0, sticky=NW, columnspan=10)

        self.image_object = ImageEnhance.Contrast(self.image).enhance(2.8)
        self.image1 = ImageTk.PhotoImage(self.image_object)
        self.canvas_img = self.canvas1.create_image(0, 0,  anchor=NW, image=self.image1)
        self.degrees_out = 0
        self.image_object_rot = self.image1
        self.rot_img = self.img
        self.image_counter = 1

        self.define_areas = None
        self.show_results = False
        self.rotated_np_img = None

        self.canvas1.bind("<ButtonPress-3>", self.start_movement)
        self.canvas1.bind("<B3-Motion>", self.move_canvas)
        self.canvas1.bind("<ButtonRelease-3>", self.stop_movement)
        self.start_x_move, self.start_y_move, self.move_x, self.move_y = 0, 0, 0, 0
        self.start_x_move_temp, self.start_y_move_temp = 0, 0
        self.end_x_move_temp, self.end_y_move_temp = 0, 0
        self.correct_x_move, self.correct_y_move = 0, 0

    def start_movement(self, event):
        self.start_x_move = self.canvas1.canvasx(event.x)
        self.start_y_move = self.canvas1.canvasy(event.y)
        self.start_x_move_temp = abs(self.canvas1.canvasx(event.x))
        self.start_y_move_temp = abs(self.canvas1.canvasy(event.y))
        self.canvas1.config(cursor="hand1")

    def move_canvas(self, event):
        self.move_x = event.x - self.start_x_move
        self.move_y = event.y - self.start_y_move
        self.canvas1.move(self.canvas_img,  int(self.move_x),  int(self.move_y))
        self.start_x_move = self.canvas1.canvasx(event.x)
        self.start_y_move = self.canvas1.canvasy(event.y)

    def stop_movement(self, event):
        self.canvas1.config(cursor="heart")
        self.end_x_move_temp = abs(int(event.x))
        self.end_y_move_temp = abs(int(event.y))
        self.correct_x_move = int(self.end_x_move_temp - self.start_x_move_temp)
        self.correct_y_move = int(self.end_y_move_temp - self.start_y_move_temp)

    @staticmethod
    def exit_processing():
        with open("status.txt", "w") as file:
            file.write("stop all")
            if os.path.isfile("config_file.json"):
                os.remove("config_file.json")
            if os.path.isfile("config_file_processed.json"):
                os.remove("config_file_processed.json")
        sys.exit()

    @staticmethod
    def need_help():
        messagebox.showinfo("Info", "1) Click the right mouse to move image to the center\n"
                                    "2) Click the left mouse to drag and draw the areas\n"
                                    "3) Start with the atrium (upper part, less bright)")

    def go_to_next_fish(self):
        if messagebox.askyesno("Next Fish", "Do you really want to go to the next fish?"):
            self.root.destroy()
            self.root.quit()
            quit()

    def rotate_image(self, degrees):
        # print(degrees)
        self.degrees_out = degrees
        new_image = self.image_object.rotate(-int(float(degrees)))
        self.rot_img = new_image
        self.image_object_rot = ImageTk.PhotoImage(new_image)
        self.canvas1.delete('all')
        self.canvas_img = self.canvas1.create_image((0, 0),
                                                    anchor=NW,
                                                    image=self.image_object_rot)
        self.canvas1.move(self.canvas_img,  int(self.correct_x_move ),  int(self.correct_y_move))
        self.canvas1.create_text(250, 40, text="Please, press <OK> to continue!",
                                 justify=LEFT, fill="white",
                                 font=("Bahnschrift", 18))

    def define_areas(self):
        self.lb.destroy()
        self.b1.destroy()
        self.play_movie_btn.destroy()
        self.s.destroy()
        self.canvas1.delete('all')
        self.lb2.grid(row=0, column=0,  pady=2, sticky=E)
        self.b2.grid(row=0, column=2, ipady=10, ipadx=10, pady=20, sticky=N)
        self.b3.grid(row=0, column=1, ipady=10, ipadx=10, pady=20, sticky=N)
        self.b4.grid(row=0, column=3, ipady=10, ipadx=10, pady=20, sticky=N)
        self.next_fish.grid(row=0, column=4, ipady=10, ipadx=10, pady=20, sticky=N)
        self.rotated_np_img = self.image.rotate(360 - float(self.degrees_out), PIL.Image.BICUBIC, expand=0)
        # self.rotated_np_img = rotate(self.img, 360 - float(self.degrees_out), reshape=False)
        self.define_areas = DefineArea(root=self.root, new_image=self.rotated_np_img)

    def reset_areas(self):
        self.define_areas.canvas.destroy()
        self.canvas1.delete('all')
        # self.rotated_np_img = rotate(self.img, 360 - float(self.degrees_out), reshape=False)
        self.rotated_np_img = self.image.rotate(360 - float(self.degrees_out), PIL.Image.BICUBIC, expand=0)
        self.define_areas = DefineArea(root=self.root, new_image=self.rotated_np_img)

    def start_progressing(self):

        if not os.path.isfile("config_file_processed.json"):
            messagebox.showerror("Area definition incomplete!", "Please define atrium and ventricle areas")
            self.reset_areas()
            return

        with open("config_file_processed.json", "r") as config_file_:
            configs_ = json.load(config_file_)
        configs_["rezise_factor"] = resize_factor_temp = self.resize_factor
        angle_ = 360 - float(self.degrees_out)
        configs_["rot_degree"] = angle_
        areas = np.asarray(configs_["heart_areas"])
        # atrium and ventricle at [(x1_atr, y1_atr), (x2_atr, y2_atr), (x1_ventr, y1_ventr), (x2_ventr, y2_ventr)]
        # defined in Class DefineAreas in self.coordinates

        self.lb2.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.b4.destroy()
        self.define_areas.canvas.destroy()
        self.canvas1.delete('all')
        self.lb3.grid(row=0, column=0,  pady=2, sticky=E)
        self.progress1.grid(row=0, column=1, ipady=5, ipadx=10, pady=20, sticky=NS)
        # self.restart.grid(row=0, column=3, ipady=5, ipadx=10, pady=20, sticky=NS)
        self.next_fish.grid(row=0, column=2, ipady=5, ipadx=10, pady=20, sticky=NS)
        self.exit_fish.grid(row=0, column=3, ipady=5, ipadx=10, pady=20, sticky=NS)

        # self.progress1_explain.grid(row=0, column=1,  pady=3, ipady=0, sticky=S)

        x_min, x_max = int(areas[0][0]), int(areas[1][0])
        y_min, y_max = int(areas[0][1]), int(areas[1][1])
        x_min2, x_max2 = int(areas[2][0]), int(areas[3][0])
        y_min2, y_max2 = int(areas[2][1]), int(areas[3][1])
        img_temp = np.asarray(self.rotated_np_img)
        img_temp2 = img_temp.copy()
        img_temp = img_temp[y_min:y_max, x_min:x_max]

        threshold = int(np.percentile(img_temp, 17.5))
        background1 = np.max(img_temp2[:, :int(0.9 * min(x_min, x_min2))])
        background2 = np.max(img_temp2[:, int(1.1 * max(x_max, x_max2)):])
        background3 = np.max(img_temp2[:int(0.9 * min(y_min, y_min2)), :])
        background4 = np.max(img_temp2[int(1.1 * max(y_max, y_max2)):, :])
        print("Take max from temporary thresholds: ", background1, background2, background3, background4, threshold)
        threshold = int(np.max([background1, background2, background3, background4, threshold])) + 1

        print("Threshold: ", threshold)
        configs_["threshold"] = threshold
        with open("config_file_processed.json", "w") as config_file_:
            json.dump(configs_, config_file_)

        def start_progress_bar():

            start_ = timeit.default_timer()
            params_temp = 0, threshold, angle_, areas, pixel_size, scale_bar_size, fish_heart_file, num_images, frames_per_second, resize_factor_temp
            process_other_images_and_plot_hearts(self.img, params_temp)
            speed = float(timeit.default_timer() - start_) / 6
            self.progress1.start(int(1000 * speed * num_images / 100))

            params = []
            for idx, image in enumerate(self.images):
                params.append((idx, threshold, angle_, areas, pixel_size,
                               scale_bar_size, fish_heart_file,  num_images, frames_per_second, resize_factor_temp
                               ))

            with mp.Pool(os.cpu_count()) as p:

                for results_ in p.starmap(process_other_images_and_plot_hearts, zip(self.images, params)):
                    atrium_values.append(results_[0])
                    atrium_area_values.append(results_[1])
                    ventricle_values.append(results_[2])
                    ventricel_area_values.append(results_[3])
                    image_container.append(results_[4])  # image in buffer

            self.canvas1.delete('all')
            img_temp_tk = PIL.Image.open(image_container[-1])
            image1 = ImageTk.PhotoImage(img_temp_tk)
            self.canvas_img = self.canvas1.create_image(int(self.w / 10), int(self.h / 10),  anchor=NW, image=image1)
            self.progress1.stop()
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(self.root.attributes, '-topmost', False)

            answer = messagebox. askyesno(
                "Completed",
            """
            All images have been analyzed.
            Heart beat curves will be created in a few seconds!
            Do you want to open heart beat curves?
            """
            )

            if answer:
                self.show_results = True

            self.root.destroy()
            self.root.quit()

        t1 = Thread(target=start_progress_bar)
        t1.daemon = True
        t1.start()
        pass


class DefineArea:
    def __init__(self, root, new_image):

        # create rotated images in canvas
        self.root = root
        # self.new_image = PIL.Image.fromarray(new_image)
        self.new_image = new_image
        display_width, display_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.display_width, self.display_height = int(display_width * 0.8), int(display_height * 0.8)
        # self.root.geometry(f'{int(display_width)}x{int(display_height)}+50+50')
        self.canvas = Canvas(self.root, cursor="heart",  height=self.display_height, width=self.display_width,
                             background="black")
        self.canvas.grid(row=1, column=0, sticky=NW, columnspan=10)
        self.new_image = ImageEnhance.Contrast(self.new_image).enhance(2.8)
        self.image1 = ImageTk.PhotoImage(self.new_image)
        self.canvas_img = self.canvas.create_image(0, 0,  anchor=NW, image=self.image1)
        self.correction_factor = 20
        self.canvas.move(self.canvas_img, 0, -int(self.display_height/self.correction_factor))
        # drag and draw rectangles
        self.x, self.y = 0, 0
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.items = list()
        self.rect = None
        self.label = None
        self.start_x = None
        self.start_y = None
        self.color = "red"
        self.label = "Atrium"
        self.coordinates = list()
        # first Atrium, second Ventricle
        # > [[start_x_a, start_y_a], [end_x_a, end_y_a], [start_x_v, start_y_v], [end_x_v, end_y_v]]
        self.counter = 0

        # move image in canvas
        self.canvas.bind("<ButtonPress-3>", self.start_movement)
        self.canvas.bind("<B3-Motion>", self.move_canvas)
        self.canvas.bind("<ButtonRelease-3>", self.stop_movement)
        self.start_x_move, self.start_y_move, self.move_x, self.move_y = 0, 0, 0, 0
        self.start_x_move_temp, self.start_y_move_temp = 0, 0
        self.end_x_move_temp, self.end_y_move_temp = 0, 0
        self.correct_x_move, self.correct_y_move = 0, 0

    def start_movement(self, event):
        self.start_x_move = self.canvas.canvasx(event.x)
        self.start_y_move = self.canvas.canvasy(event.y)
        self.start_x_move_temp = abs(self.canvas.canvasx(event.x))
        self.start_y_move_temp = abs(self.canvas.canvasy(event.y))
        self.canvas.config(cursor="hand1")

    def move_canvas(self, event):
        self.move_x = event.x - self.start_x_move
        self.move_y = event.y - self.start_y_move
        self.canvas.move(self.canvas_img,  int(self.move_x),  int(self.move_y))
        self.start_x_move = self.canvas.canvasx(event.x)
        self.start_y_move = self.canvas.canvasy(event.y)

    def stop_movement(self, event):
        self.canvas.config(cursor="heart")
        self.end_x_move_temp = abs(int(event.x))
        self.end_y_move_temp = abs(int(event.y))
        self.correct_x_move = int(self.end_x_move_temp - self.start_x_move_temp)
        self.correct_y_move = int(self.end_y_move_temp - self.start_y_move_temp)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if self.rect:
            self.color = "blue"
            self.label = "Ventricle"

        if self.counter < 2:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline=self.color, width=5)

            self.label = self.canvas.create_text(self.start_x + 60, self.start_y + 25,
                                                 justify=RIGHT,
                                                 text=self.label, fill=self.color,
                                                 font=("Bahnschrift", 16)
                                                 )
            self.items.append(self.rect)
            self.items.append(self.label)

        self.counter += 1
        """self.coordinates.append([int(self.start_x) - self.correct_x_move,
                                 int(self.start_y) - self.correct_y_move + int(self.h/self.correction_factor)
                                 ])"""

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        if self.counter < 3:
            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        # print(event.x, event.y, self.color)
        if self.counter < 3:
            coord = self.canvas.coords(self.rect)   # [x_lo, y_lo, x_ru, y_ru}

            self.coordinates.append([int(coord[0]) - self.correct_x_move,
                                     int(coord[1]) - self.correct_y_move + int(self.display_height/self.correction_factor)
                                     ])
            self.coordinates.append([int(coord[2]) - self.correct_x_move,
                                     int(coord[3]) - self.correct_y_move + int(self.display_height/self.correction_factor)
                                     ])

        if self.color == "blue":
            with open("config_file.json", "r") as config_file_:
                configs_ = json.load(config_file_)
                configs_["heart_areas"] = self.coordinates
            with open("config_file_processed.json", "w") as config_file_:
                json.dump(configs_, config_file_)
            # print(self.coordinates)
            self.canvas.create_text(250, 40,
                                    text="Please, press <OK> to start processing\n"
                                         "or <Reset> to define boxes again!",
                                    justify=LEFT,
                                    fill="white",
                                    font=("Bahnschrift", 18))


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
        vidcap = cv2.VideoCapture(self.path_in)

        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        # print("um per px: ", vidcap.get(2))
        fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)     # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count/fps)
        print("Hight (px): ", vidcap.get(4))
        print("Width (px): ", vidcap.get(3))
        print("frames per sec: ", fps)
        print("duration (sec): ", (frame_count/fps))
        print("Total number of images: ", vidcap.get(7) - 1)
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
                avi_images.append(gray)
                # plt.imshow(gray)
                # plt.show()
        if skip_images_factor > 1:
            return avi_images[::skip_images_factor], fps / skip_images_factor
        else:
            return avi_images, fps

    def extract_images_from_czi(self):
        reader = CziReader(self.path_in)
        images_temp = reader.data
        images_temp = np.asarray(images_temp)
        print(reader.dims['Y'])
        images_temp = [cv2.normalize(x[0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for x in images_temp]
        # images_temp = [np.asarray(x[0]) for x in images_temp]
        # images_temp = [[int(x) for x in y] for y in images_temp]
        # dimension == 1
        # images_temp [max(y) for y in images_temp]
        # images_temp = [PIL.Image.fromarray(x) for x in images_temp]
        # images_temp = [img.convert("L") for img in images_temp]
        # images_temp = np.asarray(images_temp)
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
            aqu_time = aqu_time.replace(",", "").replace("'", "")
            aqu_time = aqu_time.split("T")[1][:-1]
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
        return images_temp, frames_per_sec

    def extract_images_from_mp4(self):
        # under construction !!!!!!
        temp = self.path_in
        return None, None

    def extract_images_from_image_folder(self):
        img_files = sorted(glob.glob(self.path_in + f"\\*.*"))
        imgs_temp = [cv2.imread(x) for x in img_files]
        imgs_temp = [cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2GRAY) for x in imgs_temp]
        height, width = imgs_temp[0].shape
        print(height, width)
        return imgs_temp


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
    yf = np.abs(rfft(temp_values))
    xf = rfftfreq(num_data_points, 1 / frames_per_second)
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

    if np.std(values) < 0.02:
        print("no heartbeat found")
        return 0, 0, [], 0, 0
    if best_score < 0.06:
        print("no fitting function found: ", best_score)
        best_score = 0
        return 0, 0, [], 0, best_score
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


def write_heart_parameter_of_first_images_in_excel():
    """
    * reads params from config-file
    * stores heart params such as size in dictionary which transformed into a excel sheet
    :return: None
    """

    # app will create config_file
    with open("config_file_processed.json", "r") as config_file_:
        configs_ = json.load(config_file_)
        angle_ = float(configs_["rot_degree"])
        print("Rotation angle: ", angle_)
        threshold = configs_["threshold"]

    # average first 6 images
    hearts = []
    for idx1, first_imgs_temp in enumerate(first_images):
        # img_temp_one = img_temp_one[0]
        ks = 3
        kernel2 = np.ones((ks, ks), np.uint8)
        first_imgs_temp = cv2.erode(first_imgs_temp, kernel2, iterations=3)
        first_imgs_temp = cv2.dilate(first_imgs_temp, kernel2, iterations=3)
        first_imgs_temp[first_imgs_temp < threshold] = 0
        first_imgs_temp = rotate(first_imgs_temp, angle_, reshape=False)
        heart_0 = np.where(first_imgs_temp >= threshold)
        hearts.append(heart_0)

    # analyze first image and extract parameter
    a = [len(heart_0[0]) * (pixel_size ** 2) for heart_0 in hearts]
    heart_size = round(np.median(a), 1)
    value_dict["Heart_size (µm^2)"] = heart_size

    x_dist = round((np.median([np.max(heart_0[0]) - np.min(heart_0[0]) for heart_0 in hearts]) * pixel_size), 1)
    y_dist = round((np.median([np.max(heart_0[1]) - np.min(heart_0[1]) for heart_0 in hearts]) * pixel_size), 1)
    value_dict["x_distance (µm)"] = x_dist
    value_dict["y_distance (µm)"] = y_dist
    if x_dist / y_dist > 1.4:
        value_dict["Round_shape"] = 0
    else:
        value_dict["Round_shape"] = 1
    return


def process_other_images_and_plot_hearts(image, param):
    """
    * is executed in class MainWindow in multiple threads to accelerate processing
    * params include
      > num of image (idx),
      > threshold for defining the heart/background,
      > angle to rotate all images,
      > areas to define direction of the heart (e.g., atrium_up),
      > pixel size (ps_) as defined in the main window (see heart_beat_GUI.py),
      > scale bar size (sb_size),
      > name of the data file (fish_file),
      > number of total frames (n_img),
      > frames per second (fps_),
      > and resize factor (resize_x) to adjust the box sizes
      (because the size of images is adjusted in MainWindow according to the monitor/display geometry)

    * process image and find atrium / ventricle
      > rotate image
      > reduce background using cv2.erode and cv2.dilate method
      > find atrium and ventricle
      > plot atrium and ventricle boxes (area of interest) and get areas and sum of intensities

    :param image: array-like
    :param param: tuple of params

    :return: atr_intensity: int
    :return: atrium_area_: int
    :return: ventr_intensity: int
    :return: ventricle_area_: int
    :return: img_in_buf: location of figure in memory; is stored in image_container array
    """

    idx, threshold, angle, areas, ps_, sb_size, fish_file, n_img, fps_, resize_x = param
    img = image
    img_temp = np.asarray(img)

    if len(img_temp) < 200:
        resize_x = 0.5
    atr_box_size = int(np.mean([(areas[1][1] - areas[0][1]), (areas[1][0] - areas[0][0])]) * 0.42 * resize_x)
    ventrl_box_size = int(np.mean([(areas[3][1] - areas[2][1]), (areas[3][0] - areas[2][0])]) * 0.38 * resize_x)

    atrium_start = areas[0][0], areas[0][1]  # x, y
    atrium_end = areas[1][0], areas[1][1]
    atrium_center = int((atrium_start[0] + atrium_end[0]) / 2), int((atrium_start[1] + atrium_end[1]) / 2)
    ventr_start = areas[2][0], areas[2][1]
    ventr_end = areas[3][0], areas[3][1]
    ventr_center = int((ventr_start[0] + ventr_end[0]) / 2), int((ventr_start[1] + ventr_end[1]) / 2)

    # get orientation of the heart
    if abs(atrium_center[0] - ventr_center[0]) < abs(atrium_center[1] - ventr_center[1]) \
            and atrium_center[1] < ventr_center[1]:
        orientation = "atrium_up"
        angle = angle + 180
    elif abs(atrium_center[0] - ventr_center[0]) < abs(atrium_center[1] - ventr_center[1]) \
            and atrium_center[1] > ventr_center[1]:
        orientation = "atrium_down"
    elif abs(atrium_center[0] - ventr_center[0]) > abs(atrium_center[1] - ventr_center[1]) \
            and atrium_center[0] < ventr_center[0]:
        orientation = "atrium_left"
        angle = angle + 90
    elif abs(atrium_center[0] - ventr_center[0]) > abs(atrium_center[1] - ventr_center[1]) \
            and atrium_center[0] > ventr_center[0]:
        orientation = "atrium_right"
        angle = angle - 90
    else:
        orientation = "not determined"

    # process images
    img_temp = rotate(img_temp, angle, reshape=False)  # rotate image
    ks = 3   # smoothen image / clean background
    kernel2 = np.ones((ks, ks), np.uint8)
    if len(img_temp) > 300:
        img_temp = cv2.erode(img_temp, kernel2, iterations=3)
        img_temp = cv2.dilate(img_temp, kernel2, iterations=3)
    else:
        threshold = threshold * 1.2
    heart = np.where(img_temp > threshold)

    # crop image
    x_min, x_max = int(np.min(heart[1]) * 0.80), int(np.max(heart[1]) * 1.20)
    y_min, y_max = int(np.min(heart[0]) * 0.65), int(np.max(heart[0]) * 1.25)
    img_temp = img_temp[y_min:y_max, x_min:x_max]

    # find heart and atrium/ ventricle
    img_temp = np.copy(img_temp)
    img_temp[img_temp <= threshold] = 0  # set background to zero
    heart = np.where(img_temp > 0)
    heart_center = np.median(heart, axis=1)
    heart_height = np.max(heart[0]) - np.min(heart[0])
    av_channel_top = int(heart_center[0] - (heart_height/6))
    av_channel_bottom = int(heart_center[0] + (heart_height/6))
    av_channel = img_temp[av_channel_top: av_channel_bottom, :]
    # ventricle
    ventr_temp = img_temp[:int(heart_center[0] - (heart_height/6)), :]
    ventricle_top = np.min(np.where(img_temp > np.max(ventr_temp) * 0.2)[0])
    ventricle_left_border = np.min(np.where(ventr_temp > np.max(ventr_temp) * 0.35)[1])
    ventricel_right_border = np.max(np.where(ventr_temp > np.max(ventr_temp) * 0.35)[1])
    ventricle_bottom = av_channel_top + np.percentile(np.where(av_channel > np.max(av_channel) * 0.85)[0], 30)
    ventricle_height = int(ventricle_bottom) - int(ventricle_top)
    ventricle_width = int(ventricel_right_border) - int(ventricle_left_border)
    ventricle_center = [ventricle_left_border + 0.5 * ventricle_width, ventricle_top + 0.5 * ventricle_height]
    # atrium
    atrium_temp = img_temp[int(heart_center[0] + (heart_height/6)):, :]
    atrium_top = av_channel_top + np.percentile(np.where(av_channel > np.max(av_channel) * 0.70)[0], 90)
    atrium_left_border = np.min(np.where(atrium_temp > np.max(atrium_temp) * 0.30)[1])
    atrium_right_border = np.max(np.where(atrium_temp > np.max(atrium_temp) * 0.30)[1])
    atrium_bottom = np.max(heart[0])
    atrium_height = int(atrium_bottom) - int(atrium_top)
    atrium_width = int(atrium_right_border) - int(atrium_left_border)
    atrium_center = [atrium_left_border + 0.5 * atrium_width, atrium_top + 0.5 * atrium_height]

    if len(heart[0]) == 0 or ventricel_right_border == 0:
        print("no atrium/ ventricle found")
        return 0, 0, 0, 0

    # plot heart images and fill atrium_values/ ventricle values
    fig = plt.figure(idx)
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    plt.imshow(img_temp)

    # plot atrium
    atrium_area_ = int((atrium_width * ps_) * (atrium_height * ps_))
    plt.text(x=atrium_left_border * 1.02, y=atrium_bottom + (heart_height/9),
             s="Atrium (%s µm$^{2}$)" % atrium_area_, c="r", va="bottom")
    # add atrium outer box
    ax.add_patch(ax.add_patch(Rectangle((atrium_left_border,
                                         atrium_top),
                                        atrium_width, atrium_height,
                                        color="red", alpha=0.5, fill=False)))
    # add atrium inner box
    atrium_inner_box_x, atrium_inner_box_y, = (int((atrium_center[0] - atr_box_size / 2) * 1.005),
                                               int((atrium_center[1] - atr_box_size / 2) * 1.005))
    ax.add_patch(ax.add_patch(Rectangle((atrium_inner_box_x, atrium_inner_box_y),
                                        atr_box_size, atr_box_size,
                                        color="red", alpha=0.5, fill=False, linestyle="--")))

    # plot ventricle
    ventricle_area_ = int((ventricle_width * ps_) * (ventricle_height * ps_))
    plt.text(x=ventricle_left_border * 1.02, y=ventricle_top - (heart_height/9),
             s="Ventricle (%s µm$^{2}$)" % ventricle_area_, c="cyan", va="top")
    # add ventricle outer box
    ax.add_patch(ax.add_patch(Rectangle((ventricle_left_border, ventricle_top),
                                        ventricle_width, ventricle_height,
                                        color="cyan", alpha=0.5, fill=False)))
    # add ventricle inner box
    ventr_inner_box_x, ventr_inner_box_y, = int((ventricle_center[0] - ventrl_box_size / 2) * 1.01), \
                                            int((ventricle_center[1] - ventrl_box_size / 2) * 1.02)
    ax.add_patch(ax.add_patch(Rectangle((ventr_inner_box_x, ventr_inner_box_y),
                                        ventrl_box_size, ventrl_box_size,
                                        color="cyan", alpha=0.5, fill=False, linestyle="--")))

    # add scale bar
    y, x = len(img_temp), len(img_temp[0])
    if sb_size > len(img_temp[0]):
        sb_size = int(atr_box_size * 1.8) # guess scale bar size
    scale_start = (0.94 * x - sb_size)
    plt.text(x=scale_start + (sb_size/2), y=0.92 * y, s="50 µm", ha="center",
             color="white", fontdict={"size": 10, "weight": "bold"})
    plt.hlines(xmin=scale_start, xmax=scale_start + sb_size, y=0.94 * y, linewidth=2, colors="white")

    plt.axis("off")
    seconds = round(1 / fps_ * (idx + 1), 2)
    plt.title(f"{fish_file}: {idx + 1}/{n_img}, {seconds} s", fontdict={"size": 7.5})

    # plt.savefig(sub_output_path + "/%s_%s_temp.png" % (fish_heart_file, idx + 1))
    img_in_buf = fig2img(fig)  # write figure into memory
    plt.close()  # close figure otherwise memory overload

    # values for curve fitting of heartbeats > sum of intensities
    a = img_temp[atrium_inner_box_y: atrium_inner_box_y + atr_box_size,
                 atrium_inner_box_x: atrium_inner_box_x + atr_box_size]
    atr_intensity = np.sum(a)
    v = img_temp[ventr_inner_box_y: ventr_inner_box_y + ventrl_box_size,
                 ventr_inner_box_x: ventr_inner_box_x + ventrl_box_size]
    ventr_intensity = np.sum(v)
    return atr_intensity, atrium_area_, ventr_intensity, ventricle_area_, img_in_buf


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf


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
    av_block_score = round(abs(f1_atrium - f1_ventricle) + abs(f1_fft_atr - f1_fft_ventr), 2)

    ############################################################################################################
    # put values into value dictionary ###
    ############################################################################################################
    # area
    max_atrium = np.median(sorted(atrium_area_values)[-3:])
    min_atrium = np.median(sorted(atrium_area_values)[:4])
    max_ventricle = np.median(sorted(ventricel_area_values)[-3:])
    min_ventricle = np.median(sorted(ventricel_area_values)[:3])
    max_vol_atrium = np.pi * max_atrium * max_atrium**0.5
    min_vol_atrium = np.pi * max_atrium * min_atrium**0.5
    max_vol_ventr = np.pi * max_ventricle * max_ventricle**0.5
    min_vol_ventr = np.pi * min_ventricle * min_ventricle**0.5

    # heart params
    value_dict["max_atrium_area (µm^2)"] = max_atrium
    value_dict["min_atrium_area (µm^2)"] = min_atrium
    value_dict["relative atrium_contractility (%)"] = ((max_atrium - min_atrium) / max_atrium) * 100
    value_dict["atrium_ejection_fraction (µm^3)"] = max_vol_atrium - min_vol_atrium
    value_dict["max_ventricle_area (µm^2)"] = max_ventricle
    value_dict["min_ventricle_area (µm^2)"] = min_ventricle
    value_dict["relative ventricle_contractility (%)"] = ((max_ventricle - min_ventricle) / max_ventricle) * 100
    value_dict["ventricle_ejection_fraction (µm^3)"] = max_vol_ventr - min_vol_ventr
    # frequencies
    value_dict["Freq_atrium (s^-1)"] = f1_atrium
    value_dict["Freq_fft_atrium (s^-1)"] = f1_fft_atr
    value_dict["Freq_ventr (s^-1)"] = f1_ventricle
    value_dict["Freq_fft_ventr (s^-1)"] = f1_fft_ventr
    value_dict["phase_shift (s)"] = phase_shift
    value_dict["arrhythmia score (norm < 0.7)"] = arrhythmia_score
    value_dict["av_block_score (norm < 0.5)"] = av_block_score
    # fit raw heartbeat data
    value_dict["Atrium_fit_score_total"] = fit_score_all_atr_temp
    value_dict["Auto_Corr_atrium"] = auto_corr_coeff
    value_dict["Ventricle_fit_score_total"] = fit_score_all_ventr_temp
    value_dict["Auto_Corr_ventr"] = auto_corr_coeff2

    # plot heartbeat signals
    plt.figure(fish_heart_file, figsize=(10, 6))
    plt.title(fish_heart_file)
    plt.subplots_adjust(hspace=0.56)
    x_values = [i/len(ventricle_values) * cut_movie_at for i in range(1, len(ventricle_values) + 1)]

    plt.subplot2grid((3, 1), (0, 0))  # position 1 !!!!!!!!!!!
    plt.text(x=frames_per_second * cut_movie_at * 1.1, y=np.mean(atrium_values), rotation=270,
             s=fish_heart_file, fontdict={"size": 6}, ha="center", va="center")
    plt.hlines(xmin=0, xmax=max(x_values), y=np.mean(atrium_values), linestyles="--", colors="gray")
    plt.plot(x_values, atrium_values, label="raw signal", c="black", linewidth=1.5)
    plt.title("Atrium: freq=%s s$^{-1}$, fft_freq=%s s$^{-1}$" % (f1_atrium, f1_fft_atr), fontdict={"size": 10.5})
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
    plt.legend(fontsize=6, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1)
    plt.ylabel("Heartbeat (RU)", va="bottom", ha="center")

    plt.subplot2grid((3, 1), (1, 0))  # position 2 !!!!!!!!!!!
    plt.hlines(xmin=0, xmax=max(x_values), y=np.mean(ventricle_values), linestyles="--", colors="gray")
    plt.title("Ventricle: freq=%s s$^{-1}$, fft_freq=%s s$^{-1}$" % (f1_ventricle, f1_fft_ventr),
              fontdict={"size": 10.5})
    plt.plot(x_values, ventricle_values, label="raw signal", c="black", linewidth=1.5)
    if len(fitfunc_ventricle) != 0:
        plt.plot(x_values, fitfunc_ventricle, linestyle="dashed", color="red", label="best fit", linewidth=1)

    plt.legend(fontsize=6, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1)
    plt.ylabel("Heartbeat (RU)", va="bottom", ha="center")
    if len(fitfunc_ventricle) != 0:
        plot_y_min, plt_y_max = min([min(fitfunc_ventricle), min(ventricle_values)]) - .2, \
                                max([max(fitfunc_ventricle), max(ventricle_values)]) + .2
    else:
        plot_y_min, plt_y_max = min(ventricle_values) - .2, max(ventricle_values) + .2

    if plot_y_min > 0.4:
        plot_y_min = 0.2
    plt.ylim(plot_y_min, plt_y_max)

    plt.subplot2grid((3, 1), (2, 0))  # position 3 !!!!!!!!!!!
    title_text = f"Atrium vs. Ventricle (phase shift: {phase_shift} s, " \
                 f"arrhythmia score:  {arrhythmia_score}, av-block score: {av_block_score})"
    plt.title(title_text, fontdict={"size": 10.5})
    plt.plot(x_values, atrium_values_norm, label="Atrium", c="midnightblue", linewidth=1, linestyle="solid")
    plt.plot(x_values, ventricle_values_norm, label="Ventricle", c="darkorange", linewidth=1, linestyle="solid")
    plt.legend(fontsize=6, bbox_to_anchor=(1, 1), loc='upper right', framealpha=1)
    plt.ylim(min(min(atrium_values_norm), min(ventricle_values_norm)) - .2,
             max(max(atrium_values_norm), max(ventricle_values_norm)) + 0.4)
    plt.ylabel("Heartbeat (RU)")
    plt.xlabel(f"Time (s) / {frames_per_second} frames/s")

    plt.savefig(main_output_path + "/%s_frequencies_fps%s.png" % (fish_heart_file, frames_per_second))
    if image_analyzer.show_results:
        os.startfile(main_output_path + "/%s_frequencies_fps%s.png" % (fish_heart_file, frames_per_second))
    plt.close()


def create_gifs_from_images():
    # create gifs and store in project subfolder
    # fp_in = sub_output_path + "/*temp.png"
    if len(image_container) != 0:
        # img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime)]
        img, *imgs = [PIL.Image.open(f) for f in image_container]
        img.save(fp=main_output_path + "/" + "%s_heart_beads_fps%s.gif" % (fish_heart_file, frames_per_second),
                 format='GIF', append_images=imgs,
                 save_all=True,
                 duration=int(1000/frames_per_second),
                 # The display duration of each frame, in milliseconds.
                 loop=10)
        # for file in glob.glob(fp_in):
        #     os.remove(file)
        for stream in image_container:
            stream.close()  # otherwise memory overload


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
    parser.add_argument('-ff', '--file_format', type=str, help='e.g. tifs or .avi',
                        default=".avi", metavar=''),
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':

    """
    main function needs to stay here as many variables are global !
    """

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

    include_area_values = True  # if False takes intensity and area values into account; default True
    take_only_area_values = False  # if False takes intensity values into account; default False

    # initialize value_dict
    value_dict = {"Project_name": main_project,
                  "Condition": condition_folder_name}

    # extract images from folders/ videos
    images, image_container = [], []
    image_extracter = ExtractImages(movie_path)
    if file_format == ".avi":
        images, frames_per_second = image_extracter.extract_images_from_avi()
    elif file_format == ".czi":
        images, frames_per_second = image_extracter.extract_images_from_czi()
    elif file_format == ".mp4":
        images, frames_per_second = image_extracter.extract_images_from_mp4()
    elif file_format in [".tif", ".jpeg", ".png"]:
        images = image_extracter.extract_images_from_image_folder()
        frames_per_second = frames_per_second
    images_final = images[:int(cut_movie_at * frames_per_second)]
    num_images = len(images_final)
    print("Number of images: ", num_images)
    value_dict["Number_images"] = num_images

    # Then extract parameter from other images > crop at 20 s (10 frames/ s)
    atrium_values, ventricle_values, ventricel_area_values, atrium_area_values = [], [], [], []
    # rotate manually and define areas in image_analyzer in tkinter application
    all_images = np.asarray(images_final)
    first_images = np.asarray(images_final)[:6]
    img_temp_one = first_images[0]
    main_window = Tk()
    image_analyzer = MainWindow(main_window, img_temp_one, all_images)
    # process_other_images_and_plot_hearts() will be run in main window
    main_window.mainloop()
    # takes first 6 images and extract params such as heart size, shape etc.
    write_heart_parameter_of_first_images_in_excel()
    # normalize values
    atrium_values = np.asarray(atrium_values / np.max(atrium_values))
    ventricle_values = np.asarray(ventricle_values / np.max(ventricle_values))
    # save values
    np.save(sub_output_path + "/atrium.npy", np.asarray(atrium_values))
    np.save(sub_output_path + "/ventricel.npy", np.asarray(ventricle_values))
    np.save(sub_output_path + "/atrium_area.npy", atrium_area_values)
    np.save(sub_output_path + "/ventricel_area.npy", ventricel_area_values)
    # check raw data
    if np.isnan(np.sum(atrium_values)) or np.isnan(np.sum(ventricle_values)) or np.max(ventricel_area_values) > 10**6:
        print("\n##############  Image Error\n")
        quit()
    if include_area_values:
        atrium_values = include_area_values_func(atrium_values, atrium_area_values)
        ventricle_values = include_area_values_func(ventricle_values, ventricel_area_values)
    if take_only_area_values:
        atrium_values = np.asarray([(x/np.max(atrium_area_values))**2 for x in atrium_area_values])
        ventricle_values = np.asarray([(x / np.max(ventricel_area_values)) ** 2 for x in ventricel_area_values])

    if frames_per_second >= 25:  # otherwise too less data points
        atrium_values = smoothen_data(atrium_values)
        ventricle_values = smoothen_data(ventricle_values)

    # create gifs and return heartbeats as sine function
    print("process_raw_data_and_plot_heart_beat_curve")
    create_gifs_from_images()
    process_raw_data_and_plot_heart_beat_curve()

    # write config file into project subfolder
    with open("config_file_processed.json", "r") as config_file:
        configs = json.load(config_file)
        configs["num_images"] = num_images
    with open(sub_output_path + r"\config_file_processed.json", "w") as config_file:
        json.dump(configs, config_file)
    os.remove("config_file_processed.json")
    # store data in excel file
    print(value_dict)
    df = pd.DataFrame(data=value_dict, index=[0])
    df.to_excel(sub_output_path + fr"\{condition_folder_name}_results.xlsx", index=False)

