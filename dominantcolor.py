import cv2
import numpy as np
from sklearn.cluster import KMeans
#based off of the opencv tutorial code and k-means medium tutorial
def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def get_dominant_color(frame, n_clusters=3):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    clt = KMeans(n_clusters=n_clusters, n_init=10)
    clt.fit(img)
    hist = find_histogram(clt)
    dominant_color = clt.cluster_centers_[np.argmax(hist)]
    return dominant_color.astype(int)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rect_w, rect_h = 150, 150
    x1, y1 = w//2 - rect_w//2, h//2 - rect_h//2
    x2, y2 = x1 + rect_w, y1 + rect_h
    roi = frame[y1:y2, x1:x2]
    dominant_color = get_dominant_color(roi, n_clusters=3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:] = dominant_color[::-1]
    cv2.imshow("Webcam", frame)
    cv2.imshow("Dominant Color", color_patch)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
