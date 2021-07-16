#include <opencv2/opencv.hpp>
#include <vector>
int main() {
	cv::Mat img = cv::Mat(200, 300, CV_8UC3, cv::Scalar(255, 0, 0));
	/*cv::imwrite("E:/CmakeProject/test20210115_c/VS_save.jpg", img);*/
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			//img.at<cv::Vec3b>(i, j)[0] = 0; //尖括号内存放数据类型
			//img.at<cv::Vec3b>(i, j)[1] = 0;
			img.at<cv::Vec3b>(i, j)[2] = 
	}
	cv::imwrite("E:/CmakeProject/test20210115_c/VS_save1.jpg", img);
	std::vector<cv::Mat> ms; //创建矩阵
	cv::split(img, ms); //通道切割
	ms[0] = cv::Scalar(0);
	ms[1] = cv::Scalar(255);
	ms[2] = cv::Scalar(0);
	cv::merge(ms, img); //通道合并
	cv::imshow("pic",img);
	cv::waitKey();
}