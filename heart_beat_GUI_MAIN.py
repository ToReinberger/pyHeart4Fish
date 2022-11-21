"""
Author: Dr. Tobias Reinberger
Data: 07.05.2022
Version: beta 1.2 for czi movies and avi movies
"""

import os
import glob
from tkinter import *
from tkinter import messagebox
import pandas as pd
import json


if __name__ == '__main__':

    """
    config file is writen in heart_beat_GUI.py
    """

    with open("config_file.json") as config_file:
        configs = json.load(config_file)

    print(configs)
    frames_per_sec = configs["frames_per_sec"]
    pixel_size = configs["pixel_size"]  # Keyence 10x objective, 3x zoom: 0.2522 µm per pixel
    cut_movie_at = configs["cut_movie"]  # default 20 sec

    project_folder = configs["input"]
    project_name = project_folder.split("/")[-1]
    output_folder = configs["output"]
    file_format = configs["file_format"]

    file_formats = [".avi", ".czi", ".mp4", ".png", ".tif", ".jpeg"]
    file_format_ = ".avi"

    # videos
    if file_format == ".avi":
        file_format_ = ".avi"
        movies = sorted(glob.glob(fr"{project_folder}\*{file_format_}"))  # key=os.path.getmtime
    elif file_format == ".czi":
        file_format_ = ".czi"
        movies = sorted(glob.glob(fr"{project_folder}\*{file_format_}"))
    elif file_format == ".mp4":
        file_format_ = ".mp4"
        movies = sorted(glob.glob(fr"{project_folder}\*{file_format_}"))
    else:
        movies = sorted(glob.glob(fr"{project_folder}\*.*"))

    # single images
    # sort into folders per well/ fish using move_images_into_fish_folders.py
    if file_format == ".png":
        movies = sorted(glob.glob(fr"{project_folder}\*.png"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.PNG"))
    elif file_format == ".tif":
        movies = sorted(glob.glob(fr"{project_folder}\*.tif"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.TIF"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.tiff"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.TIFF"))
    elif file_format == ".jpeg":
        movies = sorted(glob.glob(fr"{project_folder}\*.jpeg"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.jpg"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.JPEG"))
        if len(movies) == 0:
            movies = sorted(glob.glob(fr"{project_folder}\*.JPG"))
    if file_format == ".png" or file_format == ".tif" or file_format == ".jpeg":
        movies = []
        for folder in os.listdir(project_folder):
            if "~" not in folder and "log" not in folder and os.path.isdir(project_folder + r"\\" + folder):
                movies.append(project_folder + r"\\" + folder)

    num_movies = len(movies)
    print("Number of conditions to analyze: ", num_movies)

    # analyze every single fish by executing heart_beat_GUI_only_one_fish_multiprocessing.py
    for idx, movie_file in enumerate(movies):
        with open("status.txt", "r") as file:
            status = file.readline()
            if "stop" in status:
                quit()  # if exist is pressed in heart_beat_GUI.py window

        print(f"{idx + 1}/{num_movies}")
        image_counter = f"{idx + 1}/{num_movies}"
        condition_folder_name = fish_heart_file = movie_file.split("\\")[-1]
        condition_folder_name = condition_folder_name.replace(f"{file_format_}", "")
        if os.path.isfile(output_folder + "\\" + condition_folder_name + fr"\{condition_folder_name}_results.xlsx") \
                and "overwrite" not in configs["overwrite_data"]:
            print(movie_file, "already analyzed")
            continue
        query = f'heart_beat_GUI_only_one_fish_multiprocessing.py "{movie_file}" ' \
                f'--output "{output_folder}" ' \
                f'--name {project_name} ' \
                f'--pixel_size {pixel_size} ' \
                f'--cut_movie_at {cut_movie_at} ' \
                f'--frames_per_sec {frames_per_sec} ' \
                f'--image_counter {image_counter} ' \
                f'{configs["overwrite_data"]} ' \
                f'--skip_images {configs["skip_images"]} ' \
                f'--file_format {file_format}'

        print(query)
        os.system(f"python {query}")  # send query to command console

    if os.path.isfile("config_file.json"):
        os.remove("config_file.json")
    # combine excel sheets
    # also executable using combine_excel_sheets.py
    combine_excel = True
    if combine_excel:
        out = []
        for movie_file in movies:
            print(movie_file)
            condition_folder_name = fish_heart_file = movie_file.split("\\")[-1]
            condition_folder_name = condition_folder_name.replace(f"{file_format_}", "").replace(",", "_").replace(" ", "_")
            out_file = output_folder + rf"\{condition_folder_name}\{condition_folder_name}_results.xlsx"
            if os.path.isfile(out_file):
                df = pd.read_excel(out_file)
                out.append(df)

        out2 = pd.concat(out)
        out2.sort_values("Condition", inplace=True)
        print("Write excel")
        out2.to_excel(output_folder + rf"\{project_name}_Final_results.xlsx", index=False)

        root2 = Tk()
        root2.lift()
        root2.attributes('-topmost', True)
        root2.after_idle(root2.attributes, '-topmost', False)
        root2.withdraw()
        answer = messagebox.askyesno("Analysis finished", "Do you want to open the excel sheet?")
        if answer:
            os.startfile(output_folder + rf"\{project_name}_Final_results.xlsx")
        root2.destroy()
        root2.mainloop()
    quit()

