from deepface import DeepFace
import os
# 设置使用GPU
# DeepFace.setting(device='cuda')
import dlib_detect_gpu
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
import pandas as pd

def face_recognition_vgg1(image_path1, image_path2):
    # 使用VGG模型进行人脸识别
    result = DeepFace.verify(img1_path=image_path1,
                             img2_path=image_path2,
                             detector_backend='opencv',
                             model_name='VGG-Face')

    if result["verified"]:
        print("人脸匹配成功！")
    else:
        print("人脸未匹配！")
    # 输出欧氏距离和余弦相似度
    print("欧氏距离:", result["distance"])
    print("余弦相似度:", result["cosine"])


def face_recognition_vgg(target_folder, train_folder):
    # 获取目标文件夹和训练集文件夹中的所有图片路径
    target_images = [os.path.join(target_folder, img) for img in os.listdir(target_folder)]
    train_images = [os.path.join(train_folder, img) for img in os.listdir(train_folder)]

    # 创建一个空的DataFrame，用于保存结果
    result_df = pd.DataFrame(columns=["Target Image", "Average Distance"])

    for target_image_path in target_images:
        # 使用VGG模型进行人脸识别，并计算与目标图片的欧氏距离
        total_distance = 0.0

        for image_path in train_images:
            result = DeepFace.verify(img1_path=target_image_path,
                                     img2_path=image_path,
                                     model_name='VGG-Face')
            if result["verified"]:
                print("人脸匹配成功！")
            else:
                print("人脸未匹配！")
            # 输出欧氏距离和余弦相似度
            print("欧氏距离:", result["distance"])
            total_distance += result["distance"]

        # 计算欧氏距离平均值
        average_distance = total_distance / len(train_images)

        # 将结果添加到DataFrame中
        result_df = result_df.append({"Target Image": os.path.basename(target_image_path),
                                      "Average Distance": average_distance},
                                     ignore_index=True)

    # 保存结果到CSV文件
    result_df.to_csv("result1.csv", index=False)


import csv

def read_csv_file(file_path):
    data_dict = {}
    values = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            filename = row[0]
            average_distance = float(row[1])
            data_dict[filename] = average_distance
            values.append(average_distance)
    return data_dict, values

def normalize_values(values):
    min_value = min(values)
    max_value = max(values)
    normalized_values = [(x - min_value) / (max_value - min_value) for x in values]
    return normalized_values

def merge_and_calculate_average(dict1, dict2, normalized_values1, normalized_values2):
    merged_dict = {}
    all_filenames = set(dict1.keys()).union(set(dict2.keys()))

    for filename in all_filenames:
        if filename in dict1 and filename in dict2:
            avg_distance = (normalized_values1[0] + normalized_values2[0]) / 2
            normalized_values1.pop(0)
            normalized_values2.pop(0)
        elif filename in dict1:
            avg_distance = normalized_values1[0]
            normalized_values1.pop(0)
        else:
            avg_distance = normalized_values2[0]
            normalized_values2.pop(0)
        merged_dict[filename] = avg_distance

    return merged_dict

def write_merged_data_to_csv(merged_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Target Image', 'Average Distance'])
        for filename, avg_distance in sorted(merged_dict.items(), key=lambda item: item[1]):
            writer.writerow([filename, avg_distance])

# if __name__ == "__main__":
#     file1 = "result.csv"
#     file2 = "result1.csv"
#     output_file = "merged_file.csv"
#
#     dict1, values1 = read_csv_file(file1)
#     dict2, values2 = read_csv_file(file2)
#
#     normalized_values1 = normalize_values(values1)
#     normalized_values2 = normalize_values(values2)
#
#     merged_dict = merge_and_calculate_average(dict1, dict2, normalized_values1, normalized_values2)
#
#     write_merged_data_to_csv(merged_dict, output_file)


if __name__ == "__main__":

    #目标文件夹和训练集文件夹路径
    target_folder = r'C:\Users\19528\Desktop\img\target'
    train_folder = r'C:\Users\19528\data\face_detect'
    face_recognition_vgg(target_folder, train_folder)



