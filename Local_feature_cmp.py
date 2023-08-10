from skimage.feature import local_binary_pattern
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import csv
import pandas as pd
import shutil

def lbp_similarity(img1, img2):
    if img1 is None or img2 is None:
        raise ValueError("One or both images were not read successfully.")
    radius = 1
    n_points = 8

    lbp1 = local_binary_pattern(img1, n_points, radius, method="uniform")
    lbp2 = local_binary_pattern(img2, n_points, radius, method="uniform")

    hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    hist1 = hist1.astype(np.float32)  # Convert to float32
    hist2 = hist2.astype(np.float32)

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)    # 直方图比较
    return similarity


def hog_similarity(img1, img2):
    hog = cv2.HOGDescriptor()
    hog1 = hog.compute(img1)
    hog2 = hog.compute(img2)
    distance = cv2.norm(hog1, hog2, cv2.NORM_L2)    # 欧式距离
    # Convert distance to similarity
    similarity = 1.0 / (1.0 + distance)
    return similarity


def sift_similarity(img1, img2):
    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()   # 暴力匹配
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)

    for match in matches:
        if len(match) == 2:  # Check if there are two matches in the list
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    return similarity


def canny_similarity(img1, img2):

    edges1 = cv2.Canny(img1, 100, 200)
    edges2 = cv2.Canny(img2, 100, 200)
    # 计算边缘图的SSIM
    similarity = ssim(edges1, edges2)

    return similarity


if __name__ == "__main1__":
    train_folder = r"C:\Users\19528\data\face_detect"
    target_folder = r"C:\Users\19528\Desktop\img\lixian1_40"
    output_csv = 'similarity_results.csv'
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # Iterate through target folder to calculate similarities
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Target', 'Max_LBP', 'Max_HOG', 'Max_SIFT', 'Max_Canny'])

        for target_filename in os.listdir(target_folder):
            target_image = cv2.imread(os.path.join(target_folder, target_filename), cv2.IMREAD_GRAYSCALE)
            # Preprocess target_image if needed

            max_lbp_sim, max_hog_sim, max_sift_sim, max_canny_sim = -1, -1, -1, -1

            for train_filename in os.listdir(train_folder):
                train_image = cv2.imread(os.path.join(train_folder, train_filename), cv2.IMREAD_GRAYSCALE)

                # 检测出人脸部分
                faces1 = face_cascade.detectMultiScale(train_image, scaleFactor=1.1, minNeighbors=5)
                faces2 = face_cascade.detectMultiScale(target_image, scaleFactor=1.1, minNeighbors=5)

                if len(faces1) > 0 and len(faces2) > 0:
                    x1, y1, w1, h1 = faces1[0]
                    x2, y2, w2, h2 = faces2[0]

                    # Extract and resize the faces
                    face1 = train_image[y1:y1 + h1, x1:x1 + w1]
                    face2 = target_image[y2:y2 + h2, x2:x2 + w2]

                    # Resize the faces to the same size
                    face1 = cv2.resize(face1, (64, 128))  # Adjust the size according needs
                    face2 = cv2.resize(face2, (64, 128))

                print(train_filename, target_filename)
                lbp_sim = lbp_similarity(face1, face2)
                hog_sim = hog_similarity(face1, face2)
                sift_sim = sift_similarity(face1, face2)
                canny_sim = canny_similarity(face1, face2)

                max_lbp_sim = max(max_lbp_sim, lbp_sim)
                max_hog_sim = max(max_hog_sim, hog_sim)
                max_sift_sim = max(max_sift_sim, sift_sim)
                max_canny_sim = max(max_canny_sim, canny_sim)

            csv_writer.writerow([target_filename, max_lbp_sim, max_hog_sim, max_sift_sim, max_canny_sim])


if __name__ == "__main1__":
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    img1 = cv2.imread(r"C:\Users\19528\data\face_detect\2.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"C:\Users\19528\data\face_detect\003EAci4gy1hegm6zmvggj610l1iw1b802.jpg", cv2.IMREAD_GRAYSCALE)

    # 检测出人脸部分
    faces1 = face_cascade.detectMultiScale(img1, scaleFactor=1.1, minNeighbors=5)
    faces2 = face_cascade.detectMultiScale(img2, scaleFactor=1.1, minNeighbors=5)

    if len(faces1) > 0 and len(faces2) > 0:
        x1, y1, w1, h1 = faces1[0]
        x2, y2, w2, h2 = faces2[0]

        # Extract and resize the faces
        face1 = img1[y1:y1 + h1, x1:x1 + w1]
        face2 = img2[y2:y2 + h2, x2:x2 + w2]

        # Resize the faces to the same size
        face1 = cv2.resize(face1, (64, 128))  # Adjust the size according needs
        face2 = cv2.resize(face2, (64, 128))
        lbp_sim = lbp_similarity(face1, face2)
        hog_sim = hog_similarity(face1, face2)
        sift_sim = sift_similarity(face1, face2)
        canny_sim = canny_similarity(face1, face2)
        print("LBP Similarity:", lbp_sim)
        print("HOG Similarity:", hog_sim)
        print("SIFT Similarity:", sift_sim)
        print("Canny Similarity:", canny_sim)


def copy_top_images(df, source_images_folder, destination_folder, num_images=40):

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    df = df[:num_images]
    # Copy the selected images to the destination folder
    for index, row in df.iterrows():
        image_name = row[df.columns[0]]
        source_path = os.path.join(source_images_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy(source_path, destination_path)
        print(f"Image {image_name} copied to {destination_folder}")


if __name__ == "__main__":

    # 读取CSV文件
    data = pd.read_csv("similarity_results.csv")
    output_base_path = r"C:\Users\19528\Desktop\img"
    # 提取照片名和变量列
    num_images = 24
    numeric_columns = ['Max_LBP','Max_HOG','Max_SIFT','Max_Canny']
    sorted_data = data.sort_values(by='Max_LBP', ascending=False)
    sorted_data.to_csv("similarity_results.csv",index=False)
    # 循环按照每个变量排序并另存为
    for column in numeric_columns:
        sorted_data = data.sort_values(by=column, ascending=False)
        # 创建文件夹
        output_folder = os.path.join(output_base_path, f'newlixian_{column}')
        csv_file_path = r"similarity_results.csv"
        source_images_folder = r"C:\Users\19528\Desktop\img\lixian1_40"
        copy_top_images(sorted_data, source_images_folder, output_folder, num_images)

    print("保存完成！")


