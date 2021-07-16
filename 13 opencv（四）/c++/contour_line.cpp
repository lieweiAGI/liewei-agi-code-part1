#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("16.jpg");
    cv::Mat gray_img, bin_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, bin_img, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> vec_4f;
    cv::findContours(bin_img, contours, vec_4f, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

//# 椭圆拟合
    cv::RotatedRect ellipse = cv::fitEllipse(contours[0]);
    cv::ellipse(img, ellipse, cv::Scalar(255, 0, 0), 2);

//# 直线拟合
    float w = img.size[0], h = img.size[1];
    cv::Vec4f line;
    cv::fitLine(contours[0], line, cv::DIST_L2, 0, 0.01, 0.01);
    float vx = line[0], vy = line[1], x = line[2], y = line[3];
    float lefty = (-x * vy / vx) + y;
    float righty = ((w - x) * vy / vx) + y;
    cv::line(img, cv::Point2f(w - 1, righty), cv::Point2f(0, lefty), cv::Scalar(0, 0, 255), 2);

    cv::imshow("img_contour", img);
    cv::waitKey(0);
}
