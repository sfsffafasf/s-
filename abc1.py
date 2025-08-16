import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from ultralytics import YOLO
from datetime import datetime
import os

class FaceRecSystem:
    def __init__(self):
        # 初始化模型和参数
        self.detector = YOLO('model/yolov8n-face-lindevs.pt')
        self.session = ort.InferenceSession('model/arcface_w600k_r50.onnx')

        # 加载特征数据库
        self.feature_db = self.load_features("features.csv")

        # 初始化考勤日志
        self.log_path = "attendance_log.csv"
        self.init_attendance_log()

    def init_attendance_log(self):
        """初始化考勤日志文件"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("timestamp,user_id,user_name\n")

    def load_features(self, csv_path):
        """加载特征数据库"""
        feature_db = {}
        if not os.path.exists(csv_path):
            print("特征数据库文件不存在！")
            return feature_db

        try:
            df = pd.read_csv(csv_path)

            # 数据清洗
            if df.isna().any().any():
                print("警告：特征数据库包含 NaN 值，正在清理...")
                df = df.dropna()
                df.to_csv(csv_path, index=False)

            for _, row in df.iterrows():
                try:
                    user_id = str(row["user_id"])
                    user_name = str(row.get("user_name", ""))

                    # 提取特征向量
                    features = row.drop(['user_id', 'user_name'] if 'user_name' in row else ['user_id']).values
                    if len(features) == 0:
                        continue

                    # 类型转换和验证
                    features = features.astype(np.float32)
                    if np.isnan(features).any():
                        print(f"用户 {user_id} 的特征包含无效值，已跳过")
                        continue

                    # 存储到数据库
                    feature_db[user_id] = {
                        'name': user_name,
                        'features': features
                    }
                except Exception as e:
                    print(f"加载用户 {user_id} 失败: {str(e)}")
                    continue

            print(f"成功加载 {len(feature_db)} 个用户特征")
            return feature_db

        except Exception as e:
            print(f"加载特征数据库失败: {str(e)}")
            return {}

    def detect_faces(self, img):
        """人脸检测"""
        results = self.detector(img, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        return bboxes

    def extract_features(self, face_img):
        """特征提取"""
        try:
            # 预处理
            face_img = cv2.resize(face_img, (112, 112))
            face_img = face_img.transpose(2, 0, 1)  # HWC to CHW
            face_img = (face_img / 255.0 - 0.5) / 0.5
            face_img = face_img.astype(np.float32)

            # ONNX推理
            inputs = {self.session.get_inputs()[0].name: np.expand_dims(face_img, 0)}
            features = self.session.run(None, inputs)[0]
            return features.flatten()
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return None

    def recognize_face(self, feature_vector, threshold=0.65):
        """人脸识别"""
        best_match = {"user_id": "Unknown", "name": "Unknown", "confidence": 0}

        if feature_vector is None:
            return best_match

        # 归一化特征向量
        feature_vector = feature_vector / np.linalg.norm(feature_vector)

        for user_id, data in self.feature_db.items():
            try:
                # 计算余弦相似度
                db_feature = data["features"]
                db_feature = db_feature / np.linalg.norm(db_feature)
                similarity = np.dot(feature_vector, db_feature)

                if similarity > best_match["confidence"]:
                    best_match = {
                        "user_id": user_id,
                        "name": data["name"],
                        "confidence": similarity
                    }
            except Exception as e:
                print(f"匹配用户 {user_id} 失败: {str(e)}")
                continue

        # 应用阈值
        if best_match["confidence"] < threshold:
            best_match = {"user_id": "Unknown", "name": "Unknown", "confidence": 0}

        return best_match

    def log_attendance(self, user_id, user_name):
        """记录考勤"""
        if user_id == "Unknown":
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a') as f:
            f.write(f"{timestamp},{user_id},{user_name}\n")


def main():
    system = FaceRecSystem()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 实时识别循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        bboxes = system.detect_faces(rgb_frame)

        # 处理每个检测到的人脸
        for (x1, y1, x2, y2) in bboxes:
            face_img = frame[y1:y2, x1:x2]

            # 特征提取和识别
            feature = system.extract_features(face_img)
            result = system.recognize_face(feature)

            # 记录考勤
            if result["user_id"] != "Unknown":
                system.log_attendance(result["user_id"], result["name"])

            # 绘制结果
            label = f"{result['name']} ({result['confidence']:.2f})"
            color = (0, 255, 0) if result["user_id"] != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 显示画面
        cv2.imshow('Face Recognition System', frame)

        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()