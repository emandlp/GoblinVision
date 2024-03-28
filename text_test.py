import pytesseract
import pyautogui
from PIL import Image
from utils.general import cv2
import os
import time
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
result_directory = "ocr_results"


def resizeImage():
    myScreenshot = pyautogui.screenshot()
    screenshot_path = os.path.join(result_directory, 'screen.png')
    myScreenshot.save(screenshot_path)
    im = Image.open(screenshot_path)  # uses PIL library to open image in memory
    left = 12
    top = 57
    right = 169
    bottom = 85
    im = im.crop((left, top, right, bottom))  # defines crop points
    im.save(os.path.join(result_directory, 'textshot.png'))  # saves new cropped image
    width, height = im.size
    new_size = (width * 4, height * 4)
    im1 = im.resize(new_size)
    im1.save(os.path.join(result_directory, 'textshot.png'))


def TestImage(preprocess, image):
    global text
    # construct the argument parse and parse the arguments
    image = cv2.imread(os.path.join(result_directory, image))
    image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove noise
    if preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)
    if preprocess == 'adaptive':
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # write the grayscale image to disk as a temporary file, so we can apply OCR to it
    timestamp = int(time.time())
    filename = "{}.png".format(timestamp)
    cv2.imwrite(os.path.join(result_directory, filename), gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
    text = pytesseract.image_to_string(Image.open(os.path.join(result_directory, filename))).strip()
    os.remove(os.path.join(result_directory, filename))
    f = open(os.path.join(result_directory, "action.txt"), "w")
    f.write(text)
    f.close()
    print(text)


def READ_Text():
    resizeImage()
    TestImage('thresh', 'textshot.png')


def main():
    while True:
        READ_Text()


if __name__ == "__main__":
    main()
