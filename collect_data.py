import pyxinput
import cv2
from mss import mss
import numpy as np
from getKeys import key_check
from utils import apply_deadzone
import time

controller = pyxinput.rController(1)
sct = mss()
mon = {'top': 31, 'left': 1, 'width': 800, 'height': 600}

data_x = []
data_y = []

counter = 1
num_files = 8
cv2.imshow("Recorded", np.zeros((300, 400, 3), np.int32))

while True:
    axes = controller.gamepad
    btns = controller.buttons
    thumb_lx = axes['thumb_lx']
    thumb_lx = float(thumb_lx) / 32768.0
    thumb_lx = float("%5.3f" % thumb_lx)

    lt = float(axes['left_trigger'])
    rt = float(axes['right_trigger'])

    trigger = (rt - lt) / 255.0
    trigger = float("%5.3f" % trigger)

    thumb_lx = apply_deadzone(thumb_lx) #thumb_lx is the steering angle between -1 and 1
    trigger = apply_deadzone(trigger) #Trigger is for both brake and acceleration(above zero is acceleration, below zero is brake)

    img = np.array(sct.grab(mon)) #converting PIL image to numpy array.
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #converting BGRA color format to RGB. Identified it as BGRA color as I was getting 4D data from Numpy array
    img = cv2.resize(img, (400, 300)) #reducing the size to 400 x 300 to reduce the training size
    keys = key_check() #check what keyboard keys are pressed
    cv2.imshow("Live", img)
    if "Y" in btns: #If controller button "Y" is pressed, then keep recording and append to data array
        cv2.putText(img, "Recording", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        joyKeys = [thumb_lx, trigger]
        data_x.append(img)
        data_y.append(joyKeys)

        counter = counter + 1
        print(counter)
    cv2.imshow("Recorded", img)
    cv2.waitKey(1)
    keys = key_check()
    btns = controller.buttons
    if "DPAD_DOWN" in btns: #When you press down button, will save the recorded video
        print("Saving!")
        time.sleep(0.5)
        num_files = num_files + 1
        np.savez("./data/data" + str(num_files) + ".npz", x=data_x, y=data_y)
        data_x = []
        data_y = []
        print("Saved! You can now continue.")
    if "DPAD_UP" in btns:
        print("Deleting!")
        time.sleep(0.5)
        data_x = []
        data_y = []