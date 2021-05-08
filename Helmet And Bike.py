import cv2

numCascade1 = cv2.CascadeClassifier("bike1.xml")
numCascade2 = cv2.CascadeClassifier("helmet6.xml")

cap = cv2.VideoCapture("bike_final.mp4")
while True:
    success,img=cap.read()
    imgResize = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
    imgGray = cv2.cvtColor(imgResize,cv2.COLOR_BGR2GRAY)
    bike = numCascade1.detectMultiScale(imgGray,1.1,5)

    for (x,y,w,h) in bike:
        area = w*h
        if area > 50:
            cv2.rectangle(imgResize,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(imgResize,(x,y-30),(x+60,y),(255,0,0),-1 )
            cv2.putText(imgResize, "Bike", (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 255), 1)
            imgROI = imgResize[y:y+h,x:x+w]
            imgROIGray = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
            helmet = numCascade2.detectMultiScale(imgROIGray, 1.1, 5)
            for (p,q,r,s) in helmet:
                area1 = r*s
                if area1 > 500:
                    cv2.rectangle(imgROI,(p,q),(p+r,q+s),(255,240,0),2)
                    cv2.rectangle(imgROI, (p, q+s+17), (p+60, q+s), (255, 240, 0), -1)
                    cv2.putText(imgROI, "Helmet",( p, q+s+17), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv2.imshow("video",imgResize,)
    if cv2.waitKey(27) == ord('q'):
        break