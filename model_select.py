import dlib
import glob
import numpy as np
import pandas as pd
import os
from skimage import io
import torch
from torchvision import transforms
from scipy.spatial import procrustes
from torch.nn import Sequential, ReLU
from torch.autograd import Variable
import shutil


def calculate_average(dictionary):    # 对字典的值进行求平均
    values = list(dictionary.values())
    if values:
        return sum(values) / len(values)
    else:
        return 0


def calculate_tarinset_descriptors(faces_folder_path):
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    # 使用预训练的dlib模型作为人脸识别模型，并移动到GPU上
    detector = dlib.get_frontal_face_detector()  # 正脸检测器（detector）用于在图像中检测出人脸的位置。它会定位出图像中可能存在的人脸区域，并返回这些区域的位置信息（通常是矩形框）
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用的设备：", device)

    candidate = []
    descriptors = []  # 存放训练集人物特征列表

    # 读取训练图像并提取特征
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("正在处理: {}".format(f))
        img = io.imread(f)
        candidate.append(f.split('\\')[-1].split('.')[0])
        # 人脸检测
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            # 使用 sp(img, d) 计算出人脸的关键点（特征点）。然后，它将图像转换为大小为 (1, 3, 224, 224) 的 torch 张量，
            # 通过对图像应用一系列的变换（调整大小为 224x224）。
            # 并将numpy数组转换为torch.Tensor
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
            img_tensor = transform(img).unsqueeze(0).to(device)
            # 人脸特征提取
            face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))  # dlib特征提取
            descriptors.append(face_descriptor)

    print('识别训练完毕！')
    return descriptors, candidate


def keypoint(target_folder_path, descriptors, candidate):
    # 人脸关键点检测器
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"

    # 使用预训练的dlib模型作为人脸识别模型，并移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用的设备：", device)
    facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    # 人脸检测器
    detector = dlib.get_frontal_face_detector()  # 正脸检测器（detector）用于在图像中检测出人脸的位置。它会定位出图像中可能存在的人脸区域，并返回这些区域的位置信息（通常是矩形框）
    sp = dlib.shape_predictor(predictor_path)

    # 使用Procrustes方法进行比对时将特征向量转换为二维特征点集
    for i, descriptor in enumerate(descriptors):
        descriptors[i] = descriptor.reshape((-1, 2))  # procrustes

    target = dict()
    # 处理待对比图片
    for f1 in glob.glob(os.path.join(target_folder_path, "*.jpg")):
        try:
            img = io.imread(f1)
            dets = detector(img, 1)
        except:
            print('输入路径有误，请检查！')
        # 将图像转换为torch.Tensor并进行预处理
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)

        if len(dets) == 0:   # 没检测到人脸
            print('Skipping image with 0 detected faces:', f1)
            continue
        if len(dets) >= 2:   # 检测到两张及以上人脸
            print('Skipping image with 2 detected faces:', f1)
            continue

        dist = []
        for k, d in enumerate(dets):
            shape = sp(img, d)
            # 现在通过模型进行人脸识别，处理预处理后的图像张量
            face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))  # dlib特征提取
            d_test = np.array(face_descriptor)
            # 在计算d_test特征向量后，将其转换为二维特征点集
            d_test = np.array(d_test).reshape((-1, 2))    # procrustes
            for i in descriptors:  # 计算距离
                mtx1, mtx2, disparity = procrustes(i, d_test)
                # dist_ = np.linalg.norm(i - d_test)  # 默认L2范式，向量归一化后计算欧式距离
                dist.append(disparity)

        # 训练集人物和距离组成一个字典
        c_d = dict(zip(candidate, dist))
        cd_sorted = sorted(c_d.items(), key=lambda d: d[1])  # sorted返回元组列表

        cd_sorted = np.array(cd_sorted)
        print("Shape of cd_sorted:", cd_sorted.shape)
        distances = cd_sorted[:, 1].astype(float)

        # 保留最像法
        min_distance = min(distances)
        print(f"min distance: {min_distance}")
        # 保存每张图片的最小距离
        target[f1.split('\\')[-1]] = min_distance

    return calculate_average(target)


def copy_top_images(csv_file_path, source_images_folder, destination_folder, num_images=40):  # 拷贝一定数量的最优图片
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Sort DataFrame by the values in the second column and select top num_images rows
    top_images = df.sort_values(by=df.columns[1])[:num_images]

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Copy the selected images to the destination folder
    for index, row in top_images.iterrows():
        image_name = row[df.columns[0]]
        source_path = os.path.join(source_images_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy(source_path, destination_path)
        print(f"Image {image_name} copied to {destination_folder}")


def create_file_dict_with_extension(directory, extension):
    file_dict = {}

    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"model directory '{directory}' does not exist.")
        return file_dict

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # 仅处理指定后缀的文件，排除子目录
        if os.path.isfile(filepath) and filename.endswith(extension):
            file_dict[filename] = 0

    return file_dict


if __name__ == "__main1__":
    # 指定目录路径和后缀
    path = r'C:\Users\19528\stable_diffusion_model'
    extension = '.safetensors'  # 模型格式

    # 创建字典
    file_dict = create_file_dict_with_extension(path, extension)

    # 打印字典
    for filename, value in file_dict.items():
        print(f"{filename}: {value}")


if __name__ == "__main__":
    # 读取有哪些模型
    path = r'C:\Users\19528\stable_diffusion_model\test'
    extension = '.safetensors'  # 模型格式
    # 创建模型-相似程度字典
    modeldict = create_file_dict_with_extension(path, extension)
    for filename, value in modeldict.items():
        print(f"{filename}: {value}")
    # 目标文件夹和训练集文件夹路径
    train_folder = r"C:\Users\19528\data\face_detect"
    basic_target_path = r"C:\Users\19528\data"
    descriptors, candidate= calculate_tarinset_descriptors(train_folder)
    for model in modeldict.keys():
        target_folder = basic_target_path + r"/" + model.split('.')[0]
        print(target_folder)
        modeldict[model] = keypoint(target_folder, descriptors, candidate)
    for filename, value in modeldict.items():
        print(f"{filename}: {value}")

    modeldict_sorted = sorted(modeldict.items(), key=lambda d: d[1])  # sorted返回元组列表 将字典转换为包含元组的列表
    df = pd.DataFrame(modeldict_sorted, columns=['模型', '距离'])  # 创建带有列名的DataFrame
    df.to_csv('model_result.csv', index=False)





