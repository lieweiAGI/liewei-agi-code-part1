#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat img1 = cv::imread("16.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("17.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat bin_img1, bin_img2;
    vector<vector<cv::Point>> contours1, contours2;
    vector<cv::Vec4i> vec_4f_1, vec_4f_2;

    cv::threshold(img1, bin_img1, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::findContours(bin_img1, contours1, vec_4f_1, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::threshold(img2, bin_img2, 127, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::findContours(bin_img2, contours2, vec_4f_2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    double ret = cv::matchShapes(contours1[0], contours2[0], cv::CONTOURS_MATCH_I2, 0.0);
    cout << ret << endl;
}
