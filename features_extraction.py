import dlib
import csv
import numpy as np
import logging
import cv2
from PIL import Image
from pathlib import Path


class FaceFeatureExtractor:
    """人脸特征提取器 - Face Feature Extractor"""

    def __init__(self):
        """初始化模型和路径 - Initialize models and paths"""
        # 数据路径 - Data paths
        self.face_images_dir = Path("data/data_faces_from_camera/")
        self.output_csv = Path("data/features_all.csv")
        self.models_dir = Path("data/data_dlib/")

        # 加载模型 - Load models
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            str(self.models_dir / "shape_predictor_68_face_landmarks.dat"))
        self.face_rec_model = dlib.face_recognition_model_v1(
            str(self.models_dir / "dlib_face_recognition_resnet_model_v1.dat"))

    def extract_face_features(self, image_path):
        """
        从单张图像中提取128D人脸特征向量 - Extract 128D face features from a single image

        Args:
            image_path: 图像文件路径 - Path to the image file

        Returns:
            face_descriptor: 人脸特征向量或0(如果未检测到人脸) - Face descriptor or 0 (if no face detected)
        """
        # 读取图像并转换颜色空间 - Read image and convert color space
        img = np.array(Image.open(image_path))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 检测人脸 - Detect faces
        faces = self.detector(img_bgr, 1)

        if not faces:
            logging.warning(f"未检测到人脸 - No face detected in {image_path}")
            return 0

        logging.info(f"检测到人脸的图像 - Image with face detected: {image_path}")

        # 提取特征 - Extract features
        shape = self.shape_predictor(img_bgr, faces[0])
        face_descriptor = self.face_rec_model.compute_face_descriptor(
            img_bgr, shape)

        return face_descriptor

    def compute_person_mean_features(self, person_dir):
        """
        计算某人多张人脸图像的平均特征向量 - Compute mean feature vector for multiple face images

        Args:
            person_dir: 包含某人多张人脸图像的目录 - Directory with multiple face images of a person

        Returns:
            features_mean: 平均特征向量 - Mean feature vector (128D)
        """
        features_list = []
        person_dir = Path(person_dir)

        # 检查目录是否存在且包含图像 - Check if directory exists and contains images
        if not person_dir.exists():
            logging.warning(f"目录不存在 - Directory doesn't exist: {person_dir}")
            return np.zeros(128, dtype=object)

        photo_files = list(person_dir.glob("*.jpg"))
        if not photo_files:
            logging.warning(f"目录中没有图像 - No images in directory: {person_dir}")
            return np.zeros(128, dtype=object)

        # 处理每张图像 - Process each image
        for photo_file in photo_files:
            logging.info(f"正在处理图像 - Processing image: {photo_file}")

            # 提取特征 - Extract features
            face_features = self.extract_face_features(photo_file)

            # 如果成功提取特征，添加到列表 - If features extracted successfully, add to list
            if face_features != 0:
                features_list.append(face_features)

        # 计算平均特征 - Compute mean features
        if features_list:
            return np.array(features_list, dtype=object).mean(axis=0)
        else:
            logging.warning(
                f"没有成功提取任何特征 - No features extracted successfully from {person_dir}")
            return np.zeros(128, dtype=object)

    def extract_all_features(self):
        """提取所有人脸特征并保存到CSV文件 - Extract all face features and save to CSV file"""
        # 获取已注册的人脸目录 - Get registered face directories
        if not self.face_images_dir.exists():
            logging.error(
                f"人脸图像目录不存在 - Face images directory doesn't exist: {self.face_images_dir}")
            return False

        person_dirs = sorted(
            [d for d in self.face_images_dir.iterdir() if d.is_dir()])
        if not person_dirs:
            logging.error("没有找到已注册的人脸 - No registered faces found")
            return False

        logging.info(
            f"找到{len(person_dirs)}个已注册人脸 - Found {len(person_dirs)} registered faces")

        # 确保输出目录存在 - Ensure output directory exists
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # 写入CSV文件 - Write to CSV file
        with open(self.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            for person_dir in person_dirs:
                # 获取人名 - Get person name
                person_name = person_dir.name
                logging.info(f"处理人脸 - Processing face: {person_name}")

                # 计算平均特征 - Compute mean features
                features_mean = self.compute_person_mean_features(person_dir)

                # 插入人名到特征向量前 - Insert person name before feature vector
                row_data = np.insert(
                    features_mean, 0, person_name.split('_')[-1])

                # 写入CSV - Write to CSV
                writer.writerow(row_data)
                logging.info(
                    f"已保存{person_name}的特征 - Features saved for {person_name}")

        logging.info(
            f"所有人脸特征已保存到 - All face features saved to: {self.output_csv}")
        return True


def main():
    """主函数 - Main function"""
    # 配置日志 - Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 创建特征提取器并执行 - Create feature extractor and run
    extractor = FaceFeatureExtractor()
    extractor.extract_all_features()


if __name__ == '__main__':
    main()
