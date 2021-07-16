#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("2.jpg");

    cv::Point2f pts1[] = {cv::Point2f(25, 30), cv::Point2f(179, 25), cv::Point2f(12, 188), cv::Point2f(189, 190)};
    cv::Point2f pts2[] = {cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(0, 200), cv::Point2f(200, 200)};

    cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);

    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, img.size());

    cv::imshow("src", img);
    cv::imshow("dst", dst_img);
    cv::waitKey(0);
}


