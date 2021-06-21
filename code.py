import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)
i=0

firstFrame = None
area = 500

while True:
	_,img = cam.read()
	text = "Normal"
	img = imutils.resize(img, width=500)
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gaussImg = cv2.GaussianBlur(grayImg,(21,21),0)
	if firstFrame is None:
		firstFrame = gaussImg
		continue
	imgDiff = cv2.absdiff(firstFrame, gaussImg)	
	threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
	threshImg = cv2.dilate(threshImg, None, iterations=2)
	cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		i+=1
		if cv2.contourArea(c) < area:
			continue
		(x,y,w,h) = cv2.boundingRect(c)
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
		text = "Moving Object detected"
		text = str(i)+''+text
		print(text)	
	cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,25),2)	


	cv2.imshow("camerafeed", img)
	cv2.imshow("thresholdImg", threshImg)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()