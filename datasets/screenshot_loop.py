import mss
import cv2
import time
import os
import pyscreenshot as ImageGrab
start_time = time.time()
display_time = 0.5

def ensure_dir():
    directory = os.path.dirname('./datasets/osrs/')
    print(directory)
    if not os.path.exists('./datasets/osrs/'):
        os.makedirs('./datasets/osrs/')

ensure_dir()

monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
title = "FPS benchmark"

sct = mss.mss()
img = 0
mob = 'cow'
while True:
    # -- include('examples/showgrabfullscreen.py') --#

    if __name__ == '__main__':
        # grab fullscreen
        im = ImageGrab.grab([0,0,1280,1024])
        # save image file
        im.save(r'./datasets/osrs/' + mob + '_' + str(img) + '.jpg', 'jpg')

        # show image in a window
        #im.show()
    # -#
    img += 1

    time.sleep(display_time)
    if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
