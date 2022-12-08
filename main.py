
import cv2 as cv
import numpy as np
import module as m
import time
from improve import adjust_brightness, adjust_dark_spots

COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 1
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0


camera = cv.VideoCapture(cameraID)
f = camera.get(cv.CAP_PROP_FPS)

width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
print(width,"x" ,height,"       FPS : " ,f)
rightmovement,leftmovement,nomovement = 0,0,0


while True:
    FRAME_COUNTER += 1
    ret, frame = camera.read()
    if ret == False:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape

    grayFrame = adjust_dark_spots(grayFrame)
    grayFrame = adjust_brightness(grayFrame)

    image, face = m.faceDetector(frame, grayFrame)
    if face is not None:

        image, PointList = m.faceLandmakDetector(frame, grayFrame, face, False)

        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]


        mask, pos, color = m.EyeTracking(frame, grayFrame, RightEyePoint)
        if pos == "Right":
            rightmovement+=1
        elif pos == "Left":
            leftmovement+=1
        else:
            nomovement+=1
        maskleft, leftPos, leftColor = m.EyeTracking(frame, grayFrame, LeftEyePoint)


        cv.putText(image, f'{pos}', (35, 95), m.fonts, 0.6, color[1], 2)
        cv.putText(image, f'{leftPos}', (int(width-140), 95),
                   m.fonts, 0.6, leftColor[1], 2)

        cv.imshow('Frame', image)
    else:
        cv.imshow('Frame', frame)

    SECONDS = time.time() - START_TIME
    FPS = FRAME_COUNTER/SECONDS

    key = cv.waitKey(1)
    if key == ord('q'):
        break


camera.release()

with open("test.txt","w") as f:
    totalmovement = leftmovement+rightmovement+nomovement
    f.write(f"Left Movement = "+str(leftmovement) + "\nNo Movement = "+ str(nomovement)+"\nRight Movement = "+str(rightmovement)+"\n\nLeft Movement "+str(round((leftmovement/totalmovement)*100,2))+"%"\
        +"\nNo Movement "+str(round((nomovement/totalmovement)*100,2))+"%"+"\nRight Movement "+str(round((rightmovement/totalmovement)*100,2))+"%")
    f.close
cv.destroyAllWindows()