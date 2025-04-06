import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import logging

# 配置日志级别
logging.basicConfig(level=logging.INFO)

# 加载Dlib模型
detector = dlib.get_frontal_face_detector()  # 人脸检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')  # 特征点检测器
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")  # 人脸识别模型


class SingleFaceRecognizer:
    """单人脸识别类，针对驾驶员监控场景优化"""
    
    def __init__(self):
        # 显示设置
        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        
        # FPS计算相关变量
        self.fps = 0
        self.fps_show = 0
        self.frame_time = 0
        self.frame_start_time = 0
        self.start_time = time.time()
        self.frame_cnt = 0
        
        # 人脸数据库
        self.features_known_list = []  # 已知人脸特征列表
        self.face_name_known_list = []  # 已知人脸名称列表
        
        # 当前帧信息
        self.current_face_rect = None  # 当前人脸矩形框
        self.current_face_name = "unknown"  # 当前人脸名称
        self.current_face_feature = None  # 当前人脸特征
        self.current_face_position = None  # 当前人脸名称显示位置
        
        # 重新识别控制
        self.reclassify_interval_cnt = 0  # 重新识别计数器
        self.reclassify_interval = 10  # 重新识别间隔帧数
        
        # 人脸检测状态
        self.face_detected = False  # 当前帧是否检测到人脸
        self.last_face_detected = False  # 上一帧是否检测到人脸
    
    def load_face_database(self):
        """从CSV文件加载已知人脸特征"""
        if not os.path.exists("data/features_all.csv"):
            logging.warning("'features_all.csv'文件不存在!")
            logging.warning("请先运行'get_faces_from_camera.py'和'features_extraction_to_csv.py'")
            return False
            
        # 读取CSV文件
        csv_rd = pd.read_csv("data/features_all.csv", header=None)
        for i in range(csv_rd.shape[0]):
            # 获取人名
            self.face_name_known_list.append(csv_rd.iloc[i][0])
            # 获取特征值
            features = []
            for j in range(1, 129):
                if csv_rd.iloc[i][j] == '':
                    features.append('0')
                else:
                    features.append(csv_rd.iloc[i][j])
            self.features_known_list.append(features)
            
        logging.info(f"数据库中的人脸数量: {len(self.features_known_list)}")
        return True
    
    def update_fps(self):
        """更新FPS计算"""
        now = time.time()
        # 每秒刷新FPS
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
    
    @staticmethod
    def calculate_distance(feature_1, feature_2):
        """计算两个特征向量间的欧氏距离"""
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        return np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    
    def draw_info(self, img):
        """在图像上绘制信息"""
        # 添加标题和统计信息
        cv2.putText(img, "DMS Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Frame: {self.frame_cnt}", (20, 80), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {self.fps_show.__round__(2)}", (20, 110), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Face: {'Detected' if self.face_detected else 'Not Detected'}", 
                    (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_face_name(self, img):
        """绘制人脸名称"""
        if self.face_detected and self.current_face_position:
            # 使用PIL绘制中文
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text(self.current_face_position, str(self.current_face_name), 
                     font=self.font_chinese, fill=(255, 255, 0))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    def identify_face(self, img, face):
        """识别人脸"""
        # 提取特征
        shape = predictor(img, face)
        face_feature = face_reco_model.compute_face_descriptor(img, shape)
        self.current_face_feature = face_feature
        
        # 计算与数据库中所有人脸的距离
        distances = []
        for i, known_feature in enumerate(self.features_known_list):
            if str(known_feature[0]) != '0.0':  # 有效特征
                distance = self.calculate_distance(face_feature, known_feature)
                distances.append(distance)
            else:  # 无效特征
                distances.append(999999999)
        
        # 找到最匹配的人脸
        if distances:
            min_distance = min(distances)
            if min_distance < 0.4:  # 阈值判断
                most_similar_idx = distances.index(min_distance)
                self.current_face_name = self.face_name_known_list[most_similar_idx]
                logging.debug(f"识别结果: {self.current_face_name}")
            else:
                self.current_face_name = "unknown"
                logging.debug("识别结果: 未知人脸")
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 检测人脸
        faces = detector(frame, 0)
        
        # 更新人脸检测状态
        self.last_face_detected = self.face_detected
        self.face_detected = len(faces) > 0
        
        # 清除当前人脸状态
        if not self.face_detected:
            self.current_face_rect = None
            self.current_face_position = None
            self.reclassify_interval_cnt = 0
            return frame
        
        # 只处理第一个人脸（假设是驾驶员）
        face = faces[0]
        self.current_face_rect = (face.left(), face.top(), face.right(), face.bottom())
        self.current_face_position = (face.left(), 
                                     int(face.bottom() + (face.bottom() - face.top()) / 4))
        
        # 绘制人脸框
        cv2.rectangle(frame, 
                     (face.left(), face.top()), 
                     (face.right(), face.bottom()), 
                     (255, 255, 255), 2)
        
        # 决定是否需要重新识别
        need_recognition = False
        
        # 情况1: 人脸首次出现
        if not self.last_face_detected:
            need_recognition = True
        
        # 情况2: 达到重新识别间隔
        elif self.current_face_name == "unknown":
            self.reclassify_interval_cnt += 1
            if self.reclassify_interval_cnt >= self.reclassify_interval:
                need_recognition = True
                self.reclassify_interval_cnt = 0
        
        # 执行人脸识别
        if need_recognition:
            self.identify_face(frame, face)
            
        # 绘制人脸名称
        frame = self.draw_face_name(frame)
        
        return frame
    
    def process(self, video_stream):
        """处理视频流"""
        if not self.load_face_database():
            return
            
        while video_stream.isOpened():
            self.frame_cnt += 1
            logging.debug(f"处理第 {self.frame_cnt} 帧")
            
            # 读取一帧
            ret, frame = video_stream.read()
            if not ret:
                break
                
            # 处理当前帧
            processed_frame = self.process_frame(frame)
            
            # 绘制信息
            self.draw_info(processed_frame)
            
            # 更新FPS
            self.update_fps()
            
            # 显示结果
            cv2.imshow("DMS Face Recognition", processed_frame)
            
            # 检查退出条件
            if cv2.waitKey(1) == ord('q'):
                break
    
    def run(self):
        """运行人脸识别程序"""
        # 从摄像头获取视频流
        cap = cv2.VideoCapture(0)
        # 处理视频流
        self.process(cap)
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


def main():
    recognizer = SingleFaceRecognizer()
    recognizer.run()


if __name__ == '__main__':
    main() 