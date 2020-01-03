import cv2
import numpy as np
import math
import dlib
import time
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
cap = cv2.VideoCapture(0)

#shanzae Zeeshan part (start)
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3

#shanzae Zeeshan part (end)

#faheel mohammad part(start)
COUNTER = 0
TOTAL = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
def play(v):
    print("OK, Playing")
    if(v):
        timer = threading.Timer(3.0,print("if loop"))
        timer.start()
    else:
        timer = threading.Timer(3.0,print("else loop"))
        timer.cancel()
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#faheel mohammad part(end)
       # landmarks = predictor(gray, face)
       # for n in range(36, 48):
        #    x = landmarks.part(n).x
        #    y = landmarks.part(n).y
         #   cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

       # k1=landmarks.part(37).x
       # k2= landmarks.part(37).y
       # o1= landmarks.part(41).x
       # o2= landmarks.part(41).y
       # cv2.line(frame,(k1,k2),(o1,o2),(1,0,255),2)
       # res=math.sqrt((k1-o1)**2 + (k2-o2)**2)
       # print(res)
#faizan nehal part(start)
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#faizan nehal part(end)		

#hamza shahid part(start)
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            #COUNTER=COUNTER+1
            #if COUNTER == 27:
            #timer = threading.Timer(3.0,play())
            #timer.start()
            #print("if")
            play(True)   
        else:
            print("else")
            #COUNTER=0
            play(False)
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

#hamza shahid part(end)