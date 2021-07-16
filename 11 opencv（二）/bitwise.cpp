#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("D:/PycharmProjects/OpenCV_test/1.jpg");
    imshow("img", img);
    Mat mask(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

    Rect r(mask.cols * 0.25, mask.rows * 0.4, 400, 100);//    Rect r(mask.cols*0.25,mask.rows*0.4,400,100);
    rectangle(mask, r, Scalar(255, 255, 255), -1);

    Mat m_out;
    bitwise_and(img, mask, m_out);

    namedWindow("img", 0);
    namedWindow("m_out", 0);
    namedWindow("mask", 0);
    imshow("img", img);
    imshow("m_out", m_out);
    imshow("mask", mask);
    waitKey(0);
}