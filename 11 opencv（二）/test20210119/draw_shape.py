import cv2
import numpy as np
import PIL
img = cv2.imread(r"1.jpg")

# cv2.line(img, (0, 0), (210, 180), color=(255,0,255), thickness=10)
# cv2.circle(img, (50, 50), 50, (0, 0, 255), 5)
# cv2.rectangle(img, (100, 30), (210, 180), color=(0, 0, 255), thickness=2)
# cv2.ellipse(img, (100, 100), (100, 50), 45, 0, 180, (255, 0, 0),0)
# pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]], np.int32)
# cv2.polylines(img, [pts], False, (0, 0, 255), 2)

cv2.putText(img, 'beautiful girl', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
cv2.imshow("pic show", img)
cv2.waitKey(0)

# a = np.random.randint(0,255,(200,300,3),np.uint8)
# img = PIL.Image.fromarray(a)
# img = img
