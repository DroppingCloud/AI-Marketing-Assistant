"""
视觉特征提取器：从原始数据中提取出模型可用的视觉特征
"""

import cv2
import numpy as np
import os
import math

# 尝试导入先进的视觉库 MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Suggest: pip install mediapipe (For better face detection accuracy)")

class VisualFeatureExtractor:
    """
    视觉特征提取器 V2.0
    
    升级说明:
    1. 人脸检测: 升级为 MediaPipe (如有) > OpenCV Haar Cascade
    2. 图像质量: 新增清晰度 (Sharpness) 检测
    3. 视觉复杂度: 新增图像熵 (Entropy) 检测
    """
    
    def __init__(self, use_mediapipe=True):
        self.mp_face_detection = None
        self.face_cascade = None
        
        # 即使 import 成功，初始化也可能失败，增加一层保护
        if HAS_MEDIAPIPE and use_mediapipe:
            try:
                # [Advanced] 初始化 MediaPipe Face Detection
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"MediaPipe init failed ({e}), falling back to OpenCV.")
                self.mp_face_detection = None
        
        # 如果 MediaPipe 不可用或初始化失败，使用 OpenCV
        if not self.mp_face_detection:
            # [Fallback] 初始化 OpenCV Haar Cascade
            # 优先使用用户指定的绝对路径
            cascade_path = "/opt/anaconda3/envs/ML/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            
            if not os.path.exists(cascade_path):
                # 如果绝对路径不存在，尝试动态查找
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                print(f"Warning: No face cascade xml found at {cascade_path}")

    def extract(self, image_path):
        """
        输入图片路径，返回包含所有视觉特征的字典
        """
        # 定义特征结构及默认值
        features = {
            # --- 光影色彩类 ---
            'brightness_mean': 0.0,    # 平均亮度 (0-1)
            'saturation_mean': 0.0,    # 平均饱和度 (0-1)
            'contrast_score': 0.0,     # 对比度 (标准差归一化)
            'colorfulness_score': 0.0, # 色彩丰富度 (Hasler & Süsstrunk)
            
            # --- 质量与风格类 ---
            'sharpness_score': 0.0,    # 清晰度 (Laplacian方差，越高越清晰)
            'entropy_score': 0.0,      # 图像熵 (衡量信息量/复杂度，越高越杂乱)
            'visual_complexity': 0.0,  # 边缘密度 
            
            # --- 内容主体类 ---
            'human_present': 0,        # 是否包含人像 (0/1)
            'face_area_ratio': 0.0,    # 人脸区域占比 (0-1)
            'face_count': 0            # 人脸数量
        }

        if not os.path.exists(image_path):
            return features
            
        # 读取图片 (BGR)
        img = cv2.imread(image_path)
        if img is None:
            return features

        try:
            height, width = img.shape[:2]
            total_pixels = height * width
            
            # 预处理：转灰度
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 预处理：转 HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # ================= 模块 A: 基础光影色彩特征 =================
            h, s, v = cv2.split(hsv_img)
            
            # [Feature] Brightness: V通道均值
            features['brightness_mean'] = np.mean(v) / 255.0
            
            # [Feature] Saturation: S通道均值
            features['saturation_mean'] = np.mean(s) / 255.0
            
            # [Feature] Contrast: 灰度图标准差
            features['contrast_score'] = np.std(gray_img) / 128.0
            
            # [Feature] Colorfulness: Hasler & Süsstrunk 算法
            # 这种算法比单纯的 Saturation 更符合人类对"色彩鲜艳"的感知
            (B, G, R) = cv2.split(img.astype("float"))
            rg = np.absolute(R - G)
            yb = np.absolute(0.5 * (R + G) - B)
            (std_rg, mean_rg) = (np.std(rg), np.mean(rg))
            (std_yb, mean_yb) = (np.std(yb), np.mean(yb))
            std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
            mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
            features['colorfulness_score'] = std_root + (0.3 * mean_root)

            # ================= 模块 B: 质量与复杂度特征 =================
            
            # [Feature] Sharpness (清晰度): Laplacian 算子的方差
            # 模糊图片边缘少，方差低；清晰图片方差高
            # 通常 > 100 算清晰，< 50 算模糊。这里做简单的数值缩放方便观察
            laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
            features['sharpness_score'] = math.log(laplacian_var + 1) # Log缩放，平滑极大值
            
            # [Feature] Visual Complexity (边缘密度): Canny
            edges = cv2.Canny(gray_img, 100, 200)
            features['visual_complexity'] = np.count_nonzero(edges) / total_pixels
            
            # [Feature] Image Entropy (图像熵): 衡量图像包含的信息量
            # 纯色图熵接近0，噪点图熵最大。可用于区分"简约风格"vs"信息密集风格"
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            logs = np.log2(hist_norm + 1e-7)
            features['entropy_score'] = -1 * (hist_norm * logs).sum()

            # ================= 模块 C: 人像/主体特征 (Advanced) =================
            
            face_area = 0.0
            face_count = 0
            
            if HAS_MEDIAPIPE and self.mp_face_detection:
                # --- Plan A: MediaPipe (Modern) ---
                # 转换颜色空间 BGR -> RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(rgb_img)
                
                if results.detections:
                    face_count = len(results.detections)
                    for detection in results.detections:
                        # 获取边界框 (relative bounding box: 0-1)
                        bboxC = detection.location_data.relative_bounding_box
                        # 累加面积 (w * h)
                        # 注意：MediaPipe 返回的是归一化的相对坐标
                        w = bboxC.width
                        h = bboxC.height
                        face_area += (w * h)
            
            elif self.face_cascade:
                # --- Plan B: Haar Cascade (Legacy) ---
                faces = self.face_cascade.detectMultiScale(
                    gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                face_count = len(faces)
                for (x, y, w, h) in faces:
                    # 像素面积 / 总像素面积
                    face_area += (w * h) / total_pixels
            
            features['human_present'] = 1 if face_count > 0 else 0
            features['face_count'] = face_count
            features['face_area_ratio'] = min(face_area, 1.0) # 限制在 1.0 以内

        except Exception as e:
            print(f"Error processing visual features for {image_path}: {e}")

        return features

# ================= 测试代码 =================
if __name__ == "__main__":
    # 图片路径
    img_path = 'image/account/brand/雅诗兰黛/68e7841a000000000302d726.jpg'
    
    # 运行提取
    extractor = VisualFeatureExtractor()
    results = extractor.extract(img_path)
    
    # 打印结果
    print("-" * 30)
    print("提取到的视觉特征:")
    for k, v in results.items():
        print(f"{k:<20}: {v:.4f}")
