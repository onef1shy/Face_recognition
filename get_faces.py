import dlib
import numpy as np
import cv2
import os
import time
import logging

# 初始化Dlib人脸检测器 - Initialize Dlib face detector
detector = dlib.get_frontal_face_detector()


class FaceRegister:
    """人脸采集与注册类 - Face registration class"""

    def __init__(self):
        # 数据存储路径 - Data storage path
        self.path_photos = "data/data_faces_from_camera/"

        # 计数器 - Counters
        self.existing_faces_cnt = 0  # 已注册人脸数 - Number of registered faces
        self.ss_cnt = 0              # 当前人脸的截图数 - Screenshot count for current face
        self.current_faces_cnt = 0   # 当前帧中的人脸数 - Faces in current frame

        # 状态标志 - Status flags
        self.save_flag = 1           # 是否可以保存图像 - Whether image can be saved
        self.press_n_flag = 0        # 是否已按N键创建新文件夹 - Whether 'N' key is pressed

        # FPS计算相关 - FPS calculation
        self.fps = 0
        self.fps_display = 0
        self.frame_time = 0
        self.frame_start_time = 0
        self.start_time = time.time()

        # 界面设置 - UI settings
        self.font = cv2.FONT_ITALIC

    def ensure_dir_exists(self):
        """确保存储目录存在 - Ensure storage directory exists"""
        if not os.path.isdir(self.path_photos):
            os.makedirs(self.path_photos, exist_ok=True)

    def get_next_face_id(self):
        """获取下一个人脸ID - Get next face ID"""
        if not os.path.isdir(self.path_photos) or not os.listdir(self.path_photos):
            return 1

        # 获取已有文件夹的最大编号 - Get maximum ID from existing folders
        person_ids = [int(p.split('_')[-1])
                      for p in os.listdir(self.path_photos)]
        return max(person_ids) + 1 if person_ids else 1

    def update_fps(self):
        """更新FPS计算 - Update FPS calculation"""
        now = time.time()
        # 每秒更新显示的FPS - Update displayed FPS every second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_display = self.fps

        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
        self.frame_start_time = now

    def draw_info(self, img):
        """在图像上绘制信息 - Draw information on image"""
        # 标题和状态信息 - Title and status info
        cv2.putText(img, "Face Register", (20, 40), self.font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {self.fps_display:.2f}",
                    (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Faces: {self.current_faces_cnt}",
                    (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # 按键指南 - Key guide
        cv2.putText(img, "N: Create face folder", (20, 350),
                    self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "S: Save current face", (20, 400),
                    self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Q: Quit", (20, 450), self.font,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def extract_face_image(self, img, face):
        """从原始图像中提取人脸区域 - Extract face region from original image"""
        # 计算扩展后的人脸区域 - Calculate expanded face region
        height = face.bottom() - face.top()
        width = face.right() - face.left()
        hh, ww = int(height/2), int(width/2)

        # 检查是否超出图像边界 - Check if face is out of image boundary
        if (face.right()+ww > 640 or face.bottom()+hh > 480 or
                face.left()-ww < 0 or face.top()-hh < 0):
            return None, False

        # 创建空白图像并复制人脸区域 - Create blank image and copy face region
        img_face = np.zeros((height*2, width*2, 3), np.uint8)
        for i in range(height*2):
            for j in range(width*2):
                img_face[i, j] = img[face.top()-hh+i, face.left()-ww+j]

        return img_face, True

    def process(self, stream):
        """处理视频流 - Process video stream"""
        # 初始化准备工作 - Initialization
        self.ensure_dir_exists()
        self.existing_faces_cnt = self.get_next_face_id() - 1
        current_face_dir = ""

        # 主循环 - Main loop
        while stream.isOpened():
            # 读取视频帧 - Read video frame
            ret, frame = stream.read()
            if not ret:
                break

            # 处理键盘输入 - Process keyboard input
            key = cv2.waitKey(1) & 0xFF

            # 检测人脸 - Detect faces
            faces = detector(frame, 0)
            self.current_faces_cnt = len(faces)

            # 处理'N'键 - 创建新的人脸文件夹 - Handle 'N' key - Create new face folder
            if key == ord('n'):
                self.existing_faces_cnt += 1
                self.ss_cnt = 0
                self.press_n_flag = 1

                current_face_dir = f"{self.path_photos}person_{self.existing_faces_cnt}"
                os.makedirs(current_face_dir, exist_ok=True)
                logging.info(
                    f"创建人脸文件夹 - Created face folder: {current_face_dir}")

            # 处理人脸 - Process faces
            for i, face in enumerate(faces):
                # 计算人脸区域 - Calculate face region
                height = face.bottom() - face.top()
                width = face.right() - face.left()
                hh, ww = int(height/2), int(width/2)

                # 检查人脸是否超出范围 - Check if face is out of range
                is_in_range = not (face.right()+ww > 640 or face.bottom()+hh > 480 or
                                   face.left()-ww < 0 or face.top()-hh < 0)

                # 绘制人脸框 - Draw face rectangle
                color = (255, 255, 255) if is_in_range else (0, 0, 255)
                cv2.rectangle(frame,
                              (face.left()-ww, face.top()-hh),
                              (face.right()+ww, face.bottom()+hh),
                              color, 2)

                # 显示超出范围警告 - Show out of range warning
                if not is_in_range:
                    cv2.putText(frame, "OUT OF RANGE", (20, 300), self.font,
                                0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    if key == ord('s'):
                        logging.warning(
                            "人脸超出范围，请调整位置 - Face out of range, please adjust position")

                # 处理'S'键 - 保存人脸 - Handle 'S' key - Save face
                if key == ord('s') and is_in_range and self.press_n_flag:
                    # 提取人脸图像 - Extract face image
                    face_img, success = self.extract_face_image(frame, face)

                    if success:
                        self.ss_cnt += 1
                        # 保存图像 - Save image
                        img_name = f"{current_face_dir}/img_face_{self.ss_cnt}.jpg"
                        cv2.imwrite(img_name, face_img)
                        logging.info(f"保存人脸图像 - Saved face image: {img_name}")
                elif key == ord('s') and not self.press_n_flag:
                    logging.warning(
                        "请先按'N'创建文件夹 - Please press 'N' first to create folder")

            # 显示信息 - Display information
            self.draw_info(frame)

            # 退出检测 - Exit detection
            if key == ord('q'):
                break

            # 更新FPS - Update FPS
            self.update_fps()

            # 显示图像 - Display image
            cv2.imshow("Face Register", frame)

    def run(self):
        """运行人脸注册程序 - Run face registration program"""
        try:
            # 初始化摄像头 - Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("无法打开摄像头 - Cannot open camera")
                return

            # 处理视频流 - Process video stream
            self.process(cap)
        finally:
            # 释放资源 - Release resources
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logging.info("程序已退出 - Program exited")


def main():
    """主函数 - Main function"""
    # 配置日志 - Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建并运行人脸注册器 - Create and run face register
    face_register = FaceRegister()
    face_register.run()


if __name__ == '__main__':
    main()
