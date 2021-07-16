import cv2
import numpy
# cap = cv2.VideoCapture(0) #摄像头
cap = cv2.VideoCapture("H:/录课视频/20210115_140046.mp4") #本地视频
#cap = cv2.VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8") #视频流
while True:
    ret, frame = cap.read() #ret:bool型判断是否接收到视频，frame:每一帧的画面
    print(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(41) & 0xFF == ord('q'): #按键退出视频播放
        break
cap.release() #释放
cv2.destroyAllWindows()
