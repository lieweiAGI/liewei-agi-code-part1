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

    // 边界矩形
    {
        cv::Rect rect = cv::boundingRect(contours[0]);
        cv::rectangle(img, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height),
                      cv::Scalar(0, 255, 0), 2);
    }
    {
        cv::RotatedRect minRect = cv::minAreaRect(contours[0]);
        cv::Point2f vs[4];
        minRect.points(vs);
        std::vector<cv::Point> contour;
        contour.push_back(vs[0]);
        contour.push_back(vs[1]);
        contour.push_back(vs[2]);
        contour.push_back(vs[3]);
        cv::polylines(img, contour, true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
    {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[0], center, radius);
        cv::circle(img, center, radius, cv::Scalar(255, 0, 0), 2);
    }


    cv::imshow("img_contour", img);
    cv::waitKey(0);
}
