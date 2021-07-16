#include <opencv2/opencv.hpp>
#include <conio.h>

int main() {
	cv::VideoCapture cap;
	cap = cv::VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8");
	while (true) {
		cv::Mat frame;
		cap >> frame;
		cv::imshow("video", frame);
		cv::waitKey(41);
		if (_kbhit()) {
			break;
		}	
	}
	cap.release();
	cv::destroyAllWindows();
}