import subprocess
import pkg_resources
import sys
import os
import time
import glob
from tkinter import messagebox, Tk


def install_packages_for_installation():
    required = {'pyshortcuts'}
    installed2 = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed2
    if missing:
        print("installing missing packages")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        time.sleep(10)


def create_shortcuts():
    installation_path = r"c:\users\*\appdata\local\programs\python\python*\lib\site-packages\pyHeart4Fish_python"
    while len(glob.glob(installation_path + "/heart_beat_GUI.py")[0]) == 0:
        time.sleep(1)
    else:
        ans = messagebox.askyesno("Create shortcut", "Do you want to create a Desktop and Start shortcut?")
        from pyshortcuts import make_shortcut
        short_cut = glob.glob(installation_path + "/heart_beat_GUI.py")[0]
        make_shortcut(short_cut,
                      name='pyHeart4Fish',
                      icon=glob.glob(installation_path + r"\Logo\PyHeart4Fish.ico")[0])
        if ans:
            print("Creating desktop short_cut")
        else:
            if os.path.isfile(fr"C:\Users\{user}\Desktop\pyHeart4Fish.lnk"):
                os.remove(fr"C:\Users\{user}\Desktop\pyHeart4Fish.lnk")

    messagebox.showinfo("Installation completed!",
                        "You can now start >pyHeart4Fish< by double clicking the desktop symbol "
                        "or Windows start > pyHeart4Fish")


if __name__ == '__main__':
    user = os.getlogin()
    install_packages_for_installation()

    root2 = Tk()
    root2.lift()
    root2.attributes('-topmost', True)
    root2.withdraw()
    create_shortcuts()
    root2.destroy()
    root2.mainloop()
    quit()
