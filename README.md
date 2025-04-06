# 人脸识别系统

基于Dlib的实时人脸检测与识别系统，支持人脸注册、特征提取和实时识别功能。

详细的实现分析与实验结果请查看我的博客文章：[人脸识别系统实现：ARM平台DMS驾驶员身份验证解决方案](https://onef1shy.github.io/2024/09/10/face_recognition/)

## 项目简介

本项目实现了一个完整的人脸识别系统，包括人脸注册、特征提取和实时识别三个主要模块。系统基于Dlib深度学习模型进行人脸检测和特征提取，采用欧氏距离匹配进行身份识别，具有较高的识别准确率和实时性能。该系统不仅可用于常规的人脸识别应用，还针对驾驶员监控系统(DMS)场景进行了优化。

## 系统架构

本项目由三个主要组件构成：

1. **人脸注册模块** (`get_faces.py`)：采集新用户的人脸图像，支持多角度采集
2. **特征提取模块** (`features_extraction.py`)：从采集的图像中提取128维特征向量
3. **人脸识别模块** (`face_reco.py`)：实时识别摄像头中的人脸，并显示身份信息

## 项目结构

```
Face_recognition/
├── data/
│   ├── data_dlib/                  # Dlib预训练模型
│   │   ├── shape_predictor_68_face_landmarks.dat
│   │   └── dlib_face_recognition_resnet_model_v1.dat
│   ├── data_faces_from_camera/     # 人脸图像存储
│   │   └── person_1/               # 每个人的人脸文件夹
│   └── features_all.csv            # 提取的人脸特征数据
├── get_faces.py                    # 人脸注册程序
├── features_extraction.py          # 特征提取程序
├── face_reco.py                    # 人脸识别程序
├── requirements.txt                # 项目依赖
├── simsun.ttc                      # 中文字体文件
└── README.md                       # 项目说明文档
```

## 技术栈

- Python 3.7+
- Dlib (用于人脸检测与特征提取)
- OpenCV (图像处理与视频流)
- NumPy (数值计算)
- Pandas (数据处理)
- Pillow (图像处理，特别是中文显示)

## 环境要求

- Python 3.7+
- 安装依赖：`pip install -r requirements.txt`
- 摄像头设备

## 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/onef1shy/Face_recognition.git
   cd Face_recognition
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 下载预训练模型
   
   系统需要两个预训练模型，应放置在 `data/data_dlib/` 目录下：
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`
   
   如果你克隆了完整仓库，这些文件应已包含。

## 使用指南

### 1. 注册新人脸

```bash
python get_faces.py
```

操作说明：
- 按 `N` 键：创建新人脸文件夹
- 按 `S` 键：保存当前人脸图像
- 按 `Q` 键：退出程序

建议为每个人采集10-20张不同角度的人脸图像，以提高识别准确率。

### 2. 提取特征

```bash
python features_extraction.py
```

此步骤将处理所有注册的人脸图像，提取特征向量并保存到 `data/features_all.csv` 文件。

### 3. 运行人脸识别

```bash
python face_reco.py
```

系统将启动摄像头，实时识别出现的人脸。

## 系统性能

系统性能主要取决于以下因素：
- 摄像头分辨率和帧率
- 计算机硬件配置
- 注册的人脸数量
- 环境光线条件

在中等配置的电脑上，系统可以实现以下性能指标：
- 人脸检测与识别速度：15-30 FPS
- 识别准确率：在良好光线条件下可达95%以上
- 识别距离：30-150厘米

## 注意事项

1. 人脸注册时，确保光线充足，面部清晰
2. 特征提取需要在注册人脸后进行
3. 最佳识别距离约为30-50厘米
4. 为提高识别准确率，请在不同光线和角度下采集人脸
5. 预训练模型文件较大，请确保有足够的存储空间

## 许可证

MIT License © [onef1shy](https://github.com/onef1shy)

## ⭐ 支持项目

欢迎 Fork 和 Star ⭐，也欢迎提出建议和PR～

## 致谢

本项目参考了 [Dlib_face_recognition_from_camera](https://github.com/coneypo/Dlib_face_recognition_from_camera) 项目的实现思路和框架，在此基础上进行了重构和优化，特别针对ARM平台和驾驶员监控系统(DMS)场景进行了改进。感谢原作者 [@coneypo](https://github.com/coneypo) 的开源贡献。 