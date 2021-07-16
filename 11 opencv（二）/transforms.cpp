#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::Mat dst;

//    cv::resize(img, dst, cv::Size(300, 300));
//    cv::transpose(img, dst);
    cv::flip(img,dst,2);

    cv::imshow("src", img);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}


