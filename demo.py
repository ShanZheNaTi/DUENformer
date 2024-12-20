# 关键点检测
import matplotlib.pyplot as plt
import cv2
import copy
plt.rcParams['figure.figsize'] = [20.0, 10.0]

#随机抽取图片
image = cv2.imread('./test/output/C60/our/58.png')
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 保留关键点
sift = cv2.xfeatures2d.SIFT_create(1000)

keypoints, descriptor = sift.detectAndCompute(training_image,None)

print("\nNumber of keypoints Detected: ", len(keypoints))
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)


cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color = (0, 255, 0),flags =cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# plt.subplot(121)
# # plt.title('FIRST')
# plt.imshow(keyp_without_size)
# plt.axis('off')  # 关闭坐标轴
plt.subplot(122)
# plt.title('SECOND')
plt.imshow(keyp_with_size)
plt.axis('off')  # 关闭坐标轴
plt.show()