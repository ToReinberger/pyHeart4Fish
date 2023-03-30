import cv2


path_in = r"C:\Users\tobia\pythonProject\pyHeart4Fish\Test_data\avi_files\fish_2.avi"
print("\nextract images from avi for ", path_in)
avi_images = []
vidcap = cv2.VideoCapture(path_in)

# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
# print("um per px: ", vidcap.get(2))
fps = round(vidcap.get(cv2.CAP_PROP_FPS), 2)     # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_count/fps)
print("Height (px): ", vidcap.get(4))
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
        cv2.imwrite(fr"C:\Users\tobia\pythonProject\pyHeart4Fish\Test_data\avi_files\fish_2_{count}.tif", image_from_avi)
        """
        print(count, end="\r")
        # print(len(image_from_avi), len(image_from_avi[0]), len(image_from_avi[0][0]))
        gray = cv2.cvtColor(image_from_avi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        # make_borders_to_image
        top, bottom, left, right = [int(width / 8)] * 4
        gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        avi_images.append(gray)"""