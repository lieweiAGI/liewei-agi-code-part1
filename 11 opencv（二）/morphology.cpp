#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("3.jpg");

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat dst;
    cv::dilate(img, dst, kernel); //膨胀
//    cv::erode(img, dst, kernel);//腐蚀
//    cv::morphologyEx(img, dst, cv::MORPH_OPEN, kernel); //开
//    cv::morphologyEx(img, dst, cv::MORPH_CLOSE, kernel); //闭
//    cv::morphologyEx(img, dst, cv::MORPH_GRADIENT, kernel); //梯度
//    cv::morphologyEx(img, dst, cv::MORPH_TOPHAT, kernel); //顶帽
//    cv::morphologyEx(img, dst, cv::MORPH_BLACKHAT, kernel); //黑帽

    cv::imshow("src", img);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}


