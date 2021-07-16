#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("15.jpg");
    cv::Mat gray_img, bin_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, bin_img, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> vec_4f;
    cv::findContours(bin_img, contours, vec_4f, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::approxPolyDP(contours.at(0), contours.at(0), 60, true);

    cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2);

    cv::imshow("img_contour", img);
    cv::waitKey(0);
}
