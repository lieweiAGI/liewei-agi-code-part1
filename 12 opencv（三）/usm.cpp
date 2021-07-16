#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::Mat dst;

    cv::GaussianBlur(img, dst, cv::Size(5, 5), 0);//高斯滤波
    cv::addWeighted(img, 2, dst, -1, 0, dst);


    cv::imshow("src", img);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}
