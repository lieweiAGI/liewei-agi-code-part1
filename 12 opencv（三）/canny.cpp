#include<opencv2/opencv.hpp>

using namespace cv;

int main() {
    cv::Mat img = cv::imread("1.jpg",IMREAD_GRAYSCALE);

    cv::GaussianBlur(img, img, cv::Size(3, 3), 1);
    cv::Canny(img, img, 50, 150);

    cv::imshow("pic show", img);
    cv::waitKey(0);
}

