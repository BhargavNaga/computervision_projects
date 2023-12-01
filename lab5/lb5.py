import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_features(sift, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return (keypoints, descriptors)

def match_features(test_descriptor, dataset_descriptor):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(test_descriptor, dataset_descriptor, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return good


dataset_images = []
for i in range(10):
    for j in range(10):
        dataset_images.append(cv2.imread(f'Images/p{i}i{j}.jpg'))

test_images = []
train_images = []
i = 0
for img in dataset_images:
    i += 1
    if (i%10 == 1) or (i%10 == 2):
        test_images.append(img)
    else:
        train_images.append(img)

sift = cv2.SIFT_create()

dataset_features = []
for img in dataset_images:
    dataset_features.append(get_features(sift, img))

test_features = []
for img in test_images:
    test_features.append(get_features(sift, img))

total_matches = 0
correct_matches = 0

for i in range(len(test_images)):
    best_matches = []
    for j in range(len(dataset_images)):
        matches = match_features(test_features[i][1], dataset_features[j][1])
        best_matches.append(len(matches))
    best_match_idx = np.argmax(best_matches)
    if best_match_idx // 10 == i // 2:
        correct_matches += 1

accuracy = (correct_matches / 20) * 100
print(f'Accuracy: {accuracy}%')

