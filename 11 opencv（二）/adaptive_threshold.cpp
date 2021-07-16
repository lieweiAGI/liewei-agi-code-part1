#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("2.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat dst1, dst2, dst3;

    cv::threshold(img, dst1, 127, 255, cv::THRESH_BINARY);
    cv::adaptiveThreshold(img, dst2, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY, 11, 2);
    cv::adaptiveThreshold(img, dst3, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 11, 2);


    cv::imshow("dst1", dst1);
    cv::imshow("dst2", dst2);
    cv::imshow("dst3", dst3);

    cv::waitKey(0);
}
