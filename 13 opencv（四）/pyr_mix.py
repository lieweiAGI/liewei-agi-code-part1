import cv2
import numpy as np

A = cv2.imread('21.jpg')
B = cv2.imread('22.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
# 边缘合成
LS = []
for i, (la, lb) in enumerate(zip(lpA, lpB)):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]

for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    # ls_ = cv2.add(ls_, LS[i])
    ls_ = cv2.addWeighted(ls_,1,LS[i],2,0)
    cv2.imshow(f"xxx{i}", ls_)

# image with direct connecting each half
real = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))

# cv2.imshow('Pyramid_blending.jpg', ls_)
# cv2.imshow('Direct_blending.jpg', real)
# print(A.shape)
# roi = real[:, 117:137]
# gas1= cv2.GaussianBlur(roi,(11,11),5)
# cv2.imshow('gasBlur', gas1)
# real[:, 117:137] = gas1
# # print(real[:, 117:137, :].shape)
# cv2.imshow("real", real)
cv2.waitKey(0)
