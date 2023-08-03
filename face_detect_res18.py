import torch
import dlib
import glob
import numpy
import os
from skimage import io
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn import Sequential, Linear, ReLU
from torch.autograd import Variable
# 人脸关键点检测器
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 训练图像文件夹
faces_folder_path = r'C:\Users\19528\data\yangmi2'


# 使用预训练的ResNet18模型作为人脸识别模型，并移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备：", device)
model = resnet18(pretrained=True)
model.fc = Linear(512, 128)  # 修改全连接层的输出维度为128
facerec = Sequential(model, ReLU()).to(device)

candidate = []         # 存放训练集人物名字
descriptors = []       # 存放训练集人物特征列表

# 人脸检测器
detector = dlib.get_frontal_face_detector()  # 正脸检测
sp = dlib.shape_predictor(predictor_path)

# 读取训练图像并提取特征
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-1].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        # 调整图像尺寸为224x224，并将numpy数组转换为torch.Tensor
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)
        # 人脸特征提取
        face_descriptor = facerec(img_tensor).cpu().detach().numpy().flatten()  # ResNet特征提取，将结果从GPU上转移到CPU上，并将其转换为NumPy数组（展开为一维向量）
        descriptors.append(face_descriptor)


print('识别训练完毕！')


# 处理待对比图片
try:
    img = io.imread(r"C:\Users\19528\data\10_yangzi(1)\10_yangzi\002j8sZsly1hdgibxteq1j60qo0w1wom02.jpg")
    dets = detector(img, 1)
except:
    print('输入路径有误，请检查！')

# 将图像转换为torch.Tensor并进行预处理
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0).to(device)

dist = []
for k, d in enumerate(dets):
    shape = sp(img, d)
    # 现在通过模型进行人脸识别，处理预处理后的图像张量
    face_descriptor = facerec(img_tensor).cpu().detach().numpy().flatten()
    d_test = numpy.array(face_descriptor)
    for i in descriptors:                # 计算距离
        dist_ = numpy.linalg.norm(i - d_test)  # 默认L2范式，向量归一化后计算欧式距离
        dist.append(dist_)

# 训练集人物和距离组成一个字典
c_d = dict(zip(candidate, dist))
cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
print("识别到的人物最有可能是: ", cd_sorted[0][0])
for key, value in cd_sorted:
    print(f"Key: {key}, Value: {value}")

