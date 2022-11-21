# pyheart4fish - a heart beat analysis tool for zebrafish #


<!README file was written according to: 
https://medium.com/analytics-vidhya/how-to-create-a-readme-md-file-8fb2e8ce24e3>

## Content ##
1. Requirements / Dependencies
2. How to run pyheart4fish - Tutorial/ Documentation
3. Troubleshooting and FAQ
4. Licensing

---
## Requirements / Dependencies ##

The pyheart4fish has been developed and tested in **windows 11**! <br/>
Supported Python versions >= 3.6. >>> https://www.python.org/downloads/

Required python packages:

* aicsimageio==4.7.0
* czifile==2019.7.2
* aicspylibczi==3.0.5
* matplotlib==3.5.1
* numpy==1.21.2
* opencv_python==4.5.5.62
* pandas==1.3.4
* Pillow==9.2.0
* scipy==1.8.0
* openpyxl==3.0.10
<br/>
```
python insall -r requirements.txt
```
---

## How to run pyheart4fish - Tutorial/ Documentation ##

### Installation ###

1. Download the complete pyheart4fish folder and run Python scripts
2. Download pyheart4fish_exe folder and run .exe file
3. use Docker image (not yet implemented)

### Run heart_beat_GUI.py ###
1. in console ``` python .../pyHeart4Fish_GUI/heart_beat_GUI.py```2
2. or right click on file > open with Python
3. run in Python IDE of your choice (e.g., Pycharm, KITE, Notepad++)

### Input data types ###
* .avi
* .czi (ZEISS file format)
* .mp4 (not supported yet! Coming soon!)
* .tif-files, .png-files or .jpg-files 
as series of frames stored in one folder per fish

### Outputs stored in the results folder ###
* Config-file as json-file
* Raw data of heart beats as numpy array
* Atrium and ventricle curves and fitting functions as plot
* Processed heart images as gif-file (10 iterations)
* Excel sheet for each fish in subfolder with all possible parameters
* Combined Excel sheet for all fish analyzed

### Test data sets in >Test_data< folder ###
* **avi_files** and avi_files_Results_example
  * frames per seconds = 29 and cut movie at = 20 sec
* **czi-files** and czi-files_Results_example
  * frames per seconds = 9.5 and cut movie at = 20 sec
* **tif_files** and tif_files_Results_example 
  * frames per seconds = 6 and cut movie at = 6 sec
    * bad example as too few frames per sec
    

### Select input and output folder and set configurations ###

![Main Window](Screenshots_tutorial/1_main_window.jpeg)


<table border="1">
  <tr>
    <td><b>Input folder</b>:</td>
    <td>contains all movie-files/ images in sub-folders for one project/ experiment </td>
   </tr>
  <tr>
     <td><b>Output folder</b>: </td>
     <td>
        > is automatically create: input + "_Results" <br/> 
        > click >Change output< 
     </td>
  </tr>
  <tr>
    <td><b> Frames per second: </b> </td>
    <td> should be > 15 for optimal results </td>
  </tr>
  <tr>
     <td> <b> Skip images: </b> </td>
     <td> > default 0, (0 - 10 possible) <br/>
          > 1 = every second frame is skipped  <br/>
          > might accelerate the process as the number of images is cut half <br/>
          > use only if frame rate is high enough! 
          </td>
  </tr>
  <tr>
     <td> <b> Pixel size: </b> </td>
     <td> > Please check the size of a pixel at your microscope <br/>
          > a wrong pixel size will give wrong heart size etc.  <br/>
          </td>
  </tr> 
  <tr>
     <td> <b> Cut movie (sec): </b> </td>
     <td> > the length of the movie should be at least 10 s <br/>
          > ensures that all movies have the same length  <br/>
          </td>
  </tr>
  <tr>
     <td> <b> File format: </b> </td>
     <td> see > <b> Input data types </b>
          </td>
  </tr>
  <tr>
     <td> <b> Overwrite data: </b> </td>
     <td> > if data need to be re-analyzed <br/>
          > if unselected all analyzed hearts will be skipped in the project folder
          </td>
  </tr>
    
</table>


By clicking ```Start``` the script ```heart_beat_GUI_MAIN.py``` is executed which iteratively <br/>
executes ```heart_beat_GUI_only_one_fish_multiprocessing.py``` and <br/>
combines all Excel sheets once all fish have been analyzed.
<br/>


### Rotate heart ###

Rotate the heart using the slider to position ventricle  at the top and atrium at the bottom.
Click ```OK``` to continue.

![Rotate](Screenshots_tutorial/2_rotate_heart_small.jpeg)

### Define atrium and ventricle ###

To distinguish between background and heart define 1) atrium and 2) ventricle area by ```Drag-and-Draw```.
Click ```OK``` to start the analysis of all frames / images.

![define_heart_areas](Screenshots_tutorial/3_define_heart_areas_small.jpeg)

The first processed images is shown after the complete analysis.
Press  ```YES``` to show heart beat curves. <br/>

The number of analyzed fish hearts is shown in the left top corner (here: 1/2 analyzed)

![progress_bar](Screenshots_tutorial/4_progress_bar.jpeg)


![show_Excel_sheet](Screenshots_tutorial/5_show_results_small.jpeg)

### Heartbeat curves ###

* **freq:** frequency / rate of heartbeats derived from fitting sine function
* **fft_freq:** frequency / rate of heartbeats derived from fast fourier transformation (FFT)
* **phase shift:** shift between atrium and ventricle
  <br/> &emsp; > A very small or very high phase shift can be a sign of arrhythmia / AV-block
* **arrhythmia score:** The lower this value, the more regular the heartbeat.
  <br/> &emsp; > a value \>0.7 is sign of arrhythmia
* **av-block score:** the absolute difference of all frequencies (freq and fft_freq) between atrium and ventricle
  <br/> &emsp; > a value \>0.5 is a sign
  
![](Screenshots_tutorial/6_heart_beat_curves._small.jpeg)


### Excel sheet for all heart movies analyzed ###

Once all fish hearts have been analyzed you can choose to open the summary Excel sheet for all fish

![](Screenshots_tutorial/7_open_excel_sheet.jpeg)

---
## Troubleshooting and FAQ ##

* make sure that the background fluorescence is as low as possible  
* In step 2, make sure you correctly select the atrium (in most case less bright part of the fish),<br/>
otherwise the background threshold is set incorrectly 
* if the heart hasn't been found correctly, please try to redefine atrium and ventricle area 

> **Please contact tobias.reinberger@uni-luebeck.de to report issues**



---
## Licensing ##

---