from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Part A
# 100 images in each category for training, the next 20 for testing

scene_categories = ['bedroom','CALsuburb','industrial','kitchen','livingroom','MITcoast','MITforest','MIThighway',
                     'MITinsidecity','MITmountain', 'MITopencountry','MITstreet','MITtallbuilding','PARoffice','store']
training = []
testing = []
TimageLabels = []
TestimageLabels = []
#taking 1st 100 images as training images from each category
for img in range(100):
    img_num=img+1
    training.insert(img, 'image_'+"{0:0=4d}".format(img_num))
#taking 20 images as testing images from 100 onwards of each category
for img in range(20):
    img_num = img + 101
    testing.insert(img, 'image_' + "{0:0=4d}".format(img_num))

# Part B
# densely sample features (SIFT descriptors) from each image
for i in range(15):
    category = scene_categories[i]
    for j in range(100):
        print('scene_categories/'+category+'/'+training[j])
        image = cv2.imread('scene_categories/'+category+'/'+training[j]+'.jpg')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        #calculate keypoints & compute the descriptors
        kp, des = sift.detectAndCompute(gray,None)
        #To write image with keypoints showing orientation of keypoints
        #img=cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imwrite('sift_keypoints.jpg',img)

      #  print(len(kp))
       # print(len(des))

# Part C
        # k-means clustering on SIFT descriptors
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 50
        ret,label,center=cv2.kmeans(des,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Part D
    # bag of words
    TimageLabels.insert(i, np.array(label).ravel())
    #print(TimageLabels[0])
    recounted = Counter(TimageLabels[i])
    print(recounted)
    f = plt.figure(i+1)
    plt.hist(TimageLabels[i], bins=20)
    plt.title('Category - '+category)
    plt.ylabel('No of Occurrences')
    plt.xlabel('Visual Words Index')
    plt.show()

# Part E
#  classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
X = TimageLabels[i]
y = TestimageLabels[i]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

# Part F
# Confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(TimageLabels[i], TestimageLabels[i]).ravel()
