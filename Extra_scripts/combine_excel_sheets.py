from tkinter import filedialog, Tk, messagebox
import glob
import os
import pandas as pd


result_folder = filedialog.askdirectory()
project_name = result_folder.split("/")[-1]
folders = glob.glob(result_folder + r"\*\*.xlsx")
print(folders)
# combine excel sheets
combine_excel = True
if combine_excel:
    out = []
    for data_file in folders:
        print(data_file)
        df = pd.read_excel(data_file)
        out.append(df)

    out2 = pd.concat(out)
    out2.sort_values("Condition", inplace=True)
    print("Write excel")
    out2.to_excel(result_folder + rf"\{project_name}_Final_results.xlsx", index=False)

    root2 = Tk()
    root2.withdraw()
    answer = messagebox.askyesno("Analysis finished", "Do you want to open the excel sheet?")
    if answer:
        os.startfile(result_folder + rf"\{project_name}_Final_results.xlsx")
    root2.destroy()
    root2.mainloop()
    quit()
