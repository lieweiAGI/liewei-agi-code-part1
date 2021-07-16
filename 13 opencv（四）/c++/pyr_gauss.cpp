#include<opencv2/opencv.hpp>

using namespace cv;

int main() {
    cv::Mat img = cv::imread("12.jpg");
    for (int i = 0; i < 3; i++) {
        cv::imshow("img" + i, img);
        cv::pyrDown(img,img);
//        cv::pyrUp(img, img);
    }

    cv::waitKey(0);
}

