//import cv2
//
//img = cv2.imread("1.jpg")
//gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
//ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
//
//cv2.imshow("gray",gray)
//cv2.imshow('binary', binary)
//cv2.waitKey(0)

#include<opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("1.jpg");

    cv::Mat gray_img;
    cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);

    cv::Mat bin_img;
    cv::threshold(gray_img,bin_img,0.,255.,cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::imshow("gray_img", gray_img);
    cv::imshow("bin_img",bin_img);
    cv::waitKey(0);
}

