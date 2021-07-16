import cv2
import numpy as np

# 传入图像和掩码区域，返回二值化图和检测后的图
def lane_detection(frame, pt):
    # 灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 创建0矩阵
    mask = np.zeros(gray.shape, dtype=np.uint8)
    # 填充指定区域形成掩码
    cv2.fillConvexPoly(mask, pt, 1)
    img = cv2.bitwise_and(gray, gray, mask=mask)
    cv2.imshow("1",img)

    _, thresh = cv2.threshold(img, 130, 145, 0)
    # 霍夫直线检测
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)
    cp = frame.copy()
    # 防止无检测直线时报错
    if lines is not None:
        # 画出检测的直线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cp, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return thresh, cp

if __name__ == '__main__':
    cap = cv2.VideoCapture("Driving Mumbai - SoBo (Downtown) [Early 2018].mp4")
    pt = np.array([[50, 540], [500, 320], [700, 320], [960, 500], [960, 540]])
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))
        # img = cv2.polylines(frame, [pt], True, (0, 0, 255), 2)
        thresh, frame = lane_detection(frame, pt)
        cv2.imshow("thresh", thresh)
        cv2.imshow("frame", frame)
        if cv2.waitKey(42) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
