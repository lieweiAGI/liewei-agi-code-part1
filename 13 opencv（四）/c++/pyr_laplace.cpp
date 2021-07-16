#include<opencv2/opencv.hpp>

using namespace cv;

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::Mat img_down, img_up, img_new;
    cv::pyrDown(img, img_down);
    cv::pyrUp(img_down, img_up);

    cv::subtract(img, img_up, img_new);

    cv::imshow("img_LP", img_new);
    cv::waitKey(0);
}

