#include<opencv2/opencv.hpp>

using namespace std;

int main() {
    cv::Mat x = (cv::Mat_<uchar>(2, 1) << 250,34);
    cv::Mat y = (cv::Mat_<uchar>(2, 1) << 10,100);

    cv::Mat addrst, subrst;
    cv::add(x, y, addrst);
    cv::subtract(x, y, subrst);

    cout << addrst << endl;
    cout << subrst << endl;
}


