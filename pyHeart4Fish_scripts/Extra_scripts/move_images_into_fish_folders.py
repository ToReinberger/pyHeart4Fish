import glob
import os
from tkinter import filedialog

project_folder = filedialog.askdirectory()
movies_temp = glob.glob(project_folder + fr"\*.*")
print("# sort images into wells/ fish folders for Aquifer systems\n", project_folder)
print(movies_temp)
# select all fishes
temp = []

for m in movies_temp:
    ident = m.split("\\")[-1].split("---")[-1].split("--")[0]
    if ident not in temp:
        print(ident)
        os.mkdir(project_folder + rf"\{ident}")
        temp.append(ident)
for ident in temp:
    print(ident)
    files = project_folder + rf"\*{ident}*.*"
    for file in glob.glob(files):
        file_name = file.split("\\")[-1]
        os.rename(file, project_folder + rf"\{ident}\{file_name}")

print("Finished")
