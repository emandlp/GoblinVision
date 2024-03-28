import pyautogui
import cv2

while True:
    top_left_x, top_left_y = pyautogui.position()
    bottom_right_x, bottom_right_y = pyautogui.position()

    # Print the coordinates of the selected area
    print("Top Left Coordinates:", (top_left_x, top_left_y))
    print("Bottom Right Coordinates:", (bottom_right_x, bottom_right_y))
    cv2.waitKey(1000)

#Top Left Coordinates: (12, 57)
#Bottom Right Coordinates: (12, 57)

#Top Left Coordinates: (169, 102)
#Bottom Right Coordinates: (169, 102)