import os
import subprocess
import pkg_resources
import sys
import time
import webbrowser
from tkinter import messagebox, Tk
import glob


def install_packages_for_installation():
    required = {'pyshortcuts'}
    installed2 = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed2
    if missing:
        print("installing missing packages")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        time.sleep(10)


def install_pyheart4fish():
    messagebox.showinfo("Download", "Please Download installation file 'pyHeart4Fish_package' from OneDrive and save in <Downloads>")
    print("Please Download installation file 'pyHeart4Fish_package' from OneDrive\nand save in <Downloads>")
    webbrowser.open("https://1drv.ms/u/s!AufmqdD8moMXg4YbDNSmpRe2pi6SWw?e=5NEZuC")

    # print(user)
    install_file = fr"C:\Users\{user}\Downloads\pyHeart4Fish-0.0.1.tar.gz"
    time.sleep(10)

    while not os.path.isfile(install_file):
        time.sleep(1)
    else:
        messagebox.showinfo("Start installation", "Download complete! Start installation")
        os.system(f"pip install {install_file}")


def create_shortcuts():
    installation_path = r"c:\users\*\appdata\local\programs\python\python*\lib\site-packages\pyHeart4Fish_scripts"
    while len(glob.glob(installation_path + "/heart_beat_GUI.py")[0]) == 0:
        time.sleep(1)
    else:
        ans = messagebox.askyesno("Create shortcut", "Do you want to create a Desktop and Start shortcut?")
        from pyshortcuts import make_shortcut
        short_cut = glob.glob(installation_path + "/heart_beat_GUI.py")[0]
        make_shortcut(short_cut,
                      name='pyHeart4Fish',
                      # icon=glob.glob(installation_path + r"\Logo\PyHeart4Fish.ico")[0]
                      )
        if ans:
            print("Creating desktop short_cut")
        else:
            if os.path.isfile(fr"C:\Users\{user}\Desktop\pyHeart4Fish.lnk"):
                os.remove(fr"C:\Users\{user}\Desktop\pyHeart4Fish.lnk")

    messagebox.showinfo("Installation completed!",
                        "You can now start >pyHeart4Fish< by double clicking the desktop symbol "
                        "or Windows start > pyHeart4Fish")


if __name__ == '__main__':

    install_packages_for_installation()

    user = os.getlogin()

    root2 = Tk()
    root2.lift()
    root2.attributes('-topmost', True)
    # root2.after_idle(root2.attributes, '-topmost', False)
    root2.withdraw()

    installed = {pkg.key for pkg in pkg_resources.working_set}
    print(installed)
    if "pyheart4fish" not in list(installed):
        install_pyheart4fish()
        create_shortcuts()
    else:
        ans_install = messagebox.askyesno("INFO",
                                          "pyHeart4Fish is already installed! Do want to re-install pyHeart4Fish?")
        if ans_install:
            install_pyheart4fish()
            create_shortcuts()

    time.sleep(1)
    if os.path.isfile(fr"C:\Users\{user}\Downloads\pyHeart4Fish-0.0.1.tar.gz"):
        os.remove(fr"C:\Users\{user}\Downloads\pyHeart4Fish-0.0.1.tar.gz")
    root2.destroy()
    root2.mainloop()
    quit()
