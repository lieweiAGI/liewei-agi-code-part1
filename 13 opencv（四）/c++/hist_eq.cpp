#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("7.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat dst;
    cv::equalizeHist(img, dst);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}
