import cv2

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_")

while True:
	ret,frame=cap.read()
	grayframe=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

	if ret==False:
		continue
	cv2.imshow('Video Frame',frame)
	cv2.imshow('Gray Video Frame',grayframe)


	#Wait for user input 'q',then you will stop the loop
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break
cap.realease()
cv2.destroyAllWindows()
