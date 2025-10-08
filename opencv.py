import numpy as np
import cv2

cap = cv2.VideoCapture(0)


lower_hsv = np.array([100, 50, 50])
upper_hsv = np.array([130, 255, 255])
lower_rgb = np.array([100, 20, 0])
upper_rgb = np.array([255, 150, 100])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)

    contours, _ = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Tracker', frame)
    # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()