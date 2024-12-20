# 特征点匹配
#%matplotlib inline
import cv2
import matplotlib.pyplot as plt
import copy
plt.rcParams['figure.figsize'] = [14.0, 7.0]

image1 = cv2.imread('./test/input/challenging-60/58.png')
image2 = cv2.imread('./test/input/challenging-60/58.png')
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(14, 7))
plt.imshow(training_image)
plt.title('left.jpg')
plt.show()

plt.imshow(query_image)
plt.title('right.jpg')
plt.show()


plt.rcParams['figure.figsize'] = [34.0, 34.0]

# training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
# query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(10000)

keypoints_train, descriptors_train = sift.detectAndCompute(training_image, None)
keypoints_query, descriptors_query = sift.detectAndCompute(query_image, None)

print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
plt.rcParams['figure.figsize'] = [10.0, 10.0]

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)

matches = bf.match(descriptors_train, descriptors_query)

result = cv2.drawMatches(training_image, keypoints_train, query_image, keypoints_query, matches, query_image, flags = 2)

print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
plt.figure(figsize=(15, 10))
plt.title('Matching Points', fontsize = 25)
plt.imshow(result)
plt.axis('off')  # 关闭坐标轴
plt.show()