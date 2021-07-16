#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::line(img, cv::Point(100, 30), cv::Point(210, 180), cv::Scalar(0, 0, 255), 2);
    cv::circle(img, cv::Point(50, 50), 30, cv::Scalar(0, 0, 255), 2);
    cv::ellipse(img, cv::Point(100, 100), cv::Point(100, 50), 0, 0, 360, cv::Scalar(0, 0, 255), 2);
    cv::rectangle(img, cv::Point(100, 30), cv::Point(210, 180), cv::Scalar(0, 0, 255), 2);

    //绘制多边形
    std::vector<cv::Point> contour;
    contour.push_back(cv::Point(10, 5));
    contour.push_back(cv::Point(50, 10));
    contour.push_back(cv::Point(70, 20));
    contour.push_back(cv::Point(20, 30));
    cv::polylines(img, contour, true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    //写字
    cv::putText(img, "beautiful girl", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv::LINE_AA);


    cv::imshow("pic show", img);
    cv::waitKey(0);
}


