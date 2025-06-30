import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully")
        cv2.imshow("Test", frame)
        cv2.waitKey(0)
    else:
        print("No frame captured")
    cap.release()
else:
    print("Camera not opened")
cv2.destroyAllWindows()