#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img = cv::imread("2.jpg");

    cv::Mat dst;
    //自定义滤波
//    cv::Mat M = (cv::Mat_<double>(3, 3) << 1, 1, 0, 1, 0, -1, 0, -1, -1);
//    cv::filter2D(img, dst, -1, M);

    //低通滤波
//    cv::blur(img, dst, cv::Size(3, 3));//均值滤波
//    cv::GaussianBlur(img, dst, cv::Size(3, 3),1,1);//高斯滤波
//    cv::medianBlur(img, dst, 3);//中值滤波
//    cv::bilateralFilter(img, dst, 9, 75, 75);//双边滤波

    //高通滤波
//    cv::Laplacian(img, dst, -1, 1); //拉普拉斯滤波

    //求梯度
//    cv::Sobel(img, dst, -1, 1, 0);
    cv::Sobel(img, dst, -1, 0, 1);
//    cv::Scharr(img, dst, -1, 1, 0);


    cv::imshow("src", img);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}
