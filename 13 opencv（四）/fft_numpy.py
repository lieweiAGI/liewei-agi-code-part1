import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('9.jpg', 0)

f = np.fft.fft2(img) #傅里叶变换
print(f)
fshift = np.fft.fftshift(f) #把中点移动到中间去
magnitude_spectrum = 20 * np.log(np.abs(fshift)) #将值取绝对值后压到0~255

plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

#去掉低频信号，留下高频信号
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

#傅里叶逆变换
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(223), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()