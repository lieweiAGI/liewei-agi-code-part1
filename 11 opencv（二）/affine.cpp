#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 50);

//    cv::Mat M = cv::getRotationMatrix2D(cv::Point(img.cols / 2, img.rows / 2), 45, 0.7);

    cv::Mat dst_img;
    cv::warpAffine(img, dst_img, M, img.size());

    cv::imshow("pic show", dst_img);
    cv::waitKey(0);
}
