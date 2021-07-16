#include "test20210115_c.h"
#include <opencv2/opencv.hpp>
using namespace std;

int main()
{
	cv::Mat img = cv::imread("pic.jpg");
	cv::imshow("picture",img);
	cv::waitKey();
}