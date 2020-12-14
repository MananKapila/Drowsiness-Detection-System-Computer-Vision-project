#!/usr/bin/env python
# coding: utf-8

# # Project by Manan Kapila 101803654 COE 29 and Akhil Rana 101803665 COE 3

# In[1]:


from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import winsound


# In[2]:


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


# In[3]:


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
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# live landmark detection
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
# calculation of MAR
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


# In[4]:



# two constants, one for the eye aspect ratio to indicate  blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 40

# similarly do the same for yawn counts
yawns = 0
yawn_thresh=4
yawn_status = False 

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
cap = cv2.VideoCapture(0)

# loop over frames from the video stream
while True:
   # grab the frame from the threaded video file stream, resize
   # it, and convert it to grayscale
   # channels)
   ret, frame = cap.read()
   frame = cv2.resize(frame, (0,0), fx=0.80, fy=0.80)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
   # detect faces in the grayscale frame
   rects = detector(gray, 0)    

   image_landmarks, lip_distance = mouth_open(frame)  
   prev_yawn_status = yawn_status    
 
   # loop over the face detections
   for rect in rects:
       # determine the facial landmarks for the face region, then
       # convert the facial landmark (x, y)-coordinates to a NumPy array
       shape = predictor(gray, rect)
       shape = face_utils.shape_to_np(shape)

       # extract the left and right eye coordinates, then use the
       # coordinates to compute the eye aspect ratio for both eyes
       leftEye = shape[lStart:lEnd]
       rightEye = shape[rStart:rEnd]
       leftEAR = eye_aspect_ratio(leftEye)
       rightEAR = eye_aspect_ratio(rightEye)

       # average the eye aspect ratio together for both eyes
       ear = (leftEAR + rightEAR) / 2.0

       # compute the convex hull for the left and right eye, then
       # visualize each of the eyes
       #EAR
       leftEyeHull = cv2.convexHull(leftEye)
       rightEyeHull = cv2.convexHull(rightEye)
       cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
       cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#-----------------------------------------------------------------------------------------------------------        
#MAR
       if lip_distance > 25:
           yawn_status = True 
           output_text = " Yawn Count: " + str(yawns + 1)
           cv2.putText(frame, output_text, (0,350),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,127),2)
       else:
           yawn_status = False 
        
       if prev_yawn_status == True and yawn_status == False:
           yawns += 1
#------------------------------------------------------------------------------------------------------------
       # check to see if the eye aspect ratio is below the blink
       # threshold, and if so, increment the blink frame counter
       if ear < EYE_AR_THRESH:
           COUNTER += 1

           # if the eyes were closed for a sufficient number of
           # then sound the alarm
           if COUNTER >= EYE_AR_CONSEC_FRAMES:
               # if the alarm is not on, turn it on
               if not ALARM_ON:
                   ALARM_ON = True
                   winsound.PlaySound("alarm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )                    
                   # check to see if an alarm file was supplied,
                   # and if so, start a thread to have the alarm
                   # sound played in the background

               # draw an alarm on the frame
               cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

       # otherwise, the eye aspect ratio is not below the blink
       # threshold, so reset the counter and alarm
       else:
           COUNTER = 0
           ALARM_ON = False
       # do the same for yawns
       if yawns>=yawn_thresh:
           cv2.putText(frame, "Subject is Yawning a lot", (0,150),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
       # draw the computed eye aspect ratio on the frame to help
       # with debugging and setting the correct eye aspect ratio
       # thresholds and frame counters
       cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

   # show the frame
   cv2.imshow('Live Landmarks', image_landmarks )
   cv2.imshow("Frame", frame)
   key = cv2.waitKey(1) & 0xFF

   # if the `q` key was pressed, break from the loop and exit
   if key == ord("q"):
       break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()


# In[ ]:




