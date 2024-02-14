from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def convertMorseToText(s):

    morseDict={
        '' : 'Check',
    '.-': 'a',
    '-...': 'b',
    '-.-.': 'c',
    '-..': 'd',
    '.': 'e',
    '..-.': 'f',
    '--.': 'g',
    '....': 'h',
    '..': 'i',
    '.---': 'j',
    '-.-': 'k',
    '.-..': 'l',
    '--': 'm',
    '-.': 'n',
    '---': 'o',
    '.--.': 'p',
    '--.-': 'q',
    '.-.': 'r',
    '...': 's',
    '-': 't',
    '..-': 'u',
    '...-': 'v',
    '.--': 'w',
    '-..-': 'x',
    '-.--': 'y',
    '--..': 'z',
    '.-.-': ' '
    }

    if morseDict.get(s)!='Check':
    	return str(morseDict.get(s))
		


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear
 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES2=6
EYE_AR_CONSEC_FRAMES3=10

COUNTER = 0
TOTAL=[]
string=""
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
fileStream = True
vs = VideoStream(src=0).start()
time.sleep(1.0)


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=550)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES and COUNTER<=EYE_AR_CONSEC_FRAMES2:
				TOTAL.append(".")
			elif COUNTER >=EYE_AR_CONSEC_FRAMES2 and COUNTER<=EYE_AR_CONSEC_FRAMES3:
				TOTAL.append("-")
			elif COUNTER>=EYE_AR_CONSEC_FRAMES3:
				s=str(convertMorseToText(''.join(TOTAL)))
				if s=="None":
					TOTAL=[]
				else:
					string+=s
					TOTAL=[]

				


			
			COUNTER = 0

		
		cv2.putText(frame, "Morse_Code: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame,"Messsage: {}".format(string), (20,100), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
 	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 	
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()