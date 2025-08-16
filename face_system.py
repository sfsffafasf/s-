import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import os

class FaceRecSystem:
    def __init__(self):
        self.detector = YOLO('model/yolov8n-face-lindevs.pt')
        self.session = ort.InferenceSession('model/arcface_w600k_r50.onnx')
        self.feature_db = self.load_features("features.csv")
        self.log_path = "attendance_log.csv"

    def load_features(self, csv_path):
        if not os.path.exists(csv_path):
            print("特征数据库文件不存在！")
            return {}
        try:
            df = pd.read_csv(csv_path)
            
            # 检查数据完整性
            if df.isna().any().any():
                print("警告：特征数据库包含 NaN 值，尝试清理...")
                df = df.dropna()  # 删除包含 NaN 的行
                df.to_csv(csv_path, index=False)  # 保存清理后的数据
                
            feature_db = {}
            for _, row in df.iterrows():
                try:
                    user_id = str(row["user_id"])
                    user_name = str(row.get("user_name", ""))
                    
                    # 提取特征并检查
                    features = row.drop(['user_id', 'user_name'] if 'user_name' in row else ['user_id']).values
                    if np.isnan(features).any():
                        print(f"警告：用户 {user_id} 的特征包含 NaN 值，跳过该用户")
                        continue
                        
                    features = features.astype(np.float32)
                    feature_db[user_id] = {
                        'name': user_name,
                        'features': features
                    }
                except Exception as e:
                    print(f"加载用户 {user_id} 的特征时出错：{str(e)}")
                    continue
                
            return feature_db
        except Exception as e:
            print(f"加载特征数据库出错：{str(e)}")
            return {}

    def detect_faces(self, img):
        results = self.detector(img, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        return bboxes

    def extract_features(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.transpose(2, 0, 1)
        face_img = (face_img / 255.0 - 0.5) / 0.5  # 归一化
        face_img = face_img.astype(np.float32)
        inputs = {self.session.get_inputs()[0].name: np.expand_dims(face_img, 0)}
        features = self.session.run(None, inputs)[0]
        return features.flatten()

    def check_in(self, frame, threshold=0.4):
        try:
            bboxes = self.detect_faces(frame)
            if len(bboxes) == 0:
                print("未检测到人脸")
                return None, frame, None

            for box in bboxes:
                x1, y1, x2, y2 = box
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                try:
                    current_feature = self.extract_features(face_img)
                    if np.isnan(current_feature).any():
                        print("警告：提取的特征包含 NaN 值")
                        continue
                        
                    best_match = None
                    best_similarity = 0
                    
                    for uid, user_data in self.feature_db.items():
                        try:
                            db_feature = user_data['features']
                            if np.isnan(db_feature).any():
                                print(f"警告：用户 {uid} 的特征包含 NaN 值，跳过比对")
                                continue
                                
                            similarity = cosine_similarity([current_feature], [db_feature])[0][0]
                            print(f"与用户 {uid} ({user_data['name']}) 的相似度: {similarity}")
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = uid
                        except Exception as e:
                            print(f"比对失败 {uid}: {str(e)}")
                            continue

                    if best_match and best_similarity > threshold:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.log_attendance(best_match, timestamp)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        user_name = self.feature_db[best_match]['name']
                        cv2.putText(frame, f"{best_similarity:.2f}",
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        print(f"[{timestamp}] 考勤成功：{best_match} ({user_name}), 相似度：{best_similarity}")
                        return best_match, frame, user_name
                    else:
                        print(f"相似度不足：{best_similarity}")
                except Exception as e:
                    print(f"特征提取失败：{str(e)}")
                    continue

            return None, frame, None
        except Exception as e:
            print(f"识别出错：{str(e)}")
            return None, frame, None

    def log_attendance(self, user_id, timestamp):
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
        else:
            df = pd.DataFrame(columns=["user_id", "timestamp"])

        new_entry = pd.DataFrame([[user_id, timestamp]], columns=df.columns)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.log_path, index=False)
        print(f"考勤记录存入日志：{user_id}, {timestamp}")

    def save_features(self, user_id, user_name, features):
        try:
            print(f"开始保存用户特征：{user_id}, {user_name}")
            
            # 1. 检查并处理特征向量
            if np.isnan(features).any():
                print("警告：特征向量包含 NaN 值")
                return False
            
            features = features.astype(np.float32)
            
            # 2. 创建新用户的数据行
            feature_list = features.tolist()  # 转换特征为列表
            data = {'user_id': [user_id], 'user_name': [user_name]}
            # 为每个特征值创建一个列
            for i, value in enumerate(feature_list):
                data[f'feature_{i}'] = [value]
            
            # 3. 创建新用户的DataFrame
            new_user_df = pd.DataFrame(data)
            
            # 4. 处理已有数据
            if os.path.exists("features.csv"):
                try:
                    existing_df = pd.read_csv("features.csv")
                    print(f"现有用户：{existing_df['user_id'].tolist()}")
                    
                    # 如果用户已存在，更新信息
                    if str(user_id) in existing_df["user_id"].astype(str).values:
                        # 获取要更新的行的索引
                        idx = existing_df[existing_df["user_id"].astype(str) == str(user_id)].index[0]
                        # 更新该行的所有值
                        for col in new_user_df.columns:
                            existing_df.at[idx, col] = new_user_df.iloc[0][col]
                        print(f"更新用户 {user_id} 的信息")
                    else:
                        # 添加新用户
                        existing_df = pd.concat([existing_df, new_user_df], ignore_index=True)
                        print(f"添加新用户 {user_id}")
                    
                    # 保存更新后的数据
                    existing_df.to_csv("features.csv", index=False)
                except Exception as e:
                    print(f"更新特征文件时出错：{str(e)}")
                    return False
            else:
                # 如果文件不存在，直接保存新文件
                new_user_df.to_csv("features.csv", index=False)
                print("创建新的特征文件")
            
            # 5. 更新内存中的特征库
            self.feature_db[str(user_id)] = {
                'name': user_name,
                'features': features
            }
            print(f"特征保存成功：{user_id}")
            return True
        
        except Exception as e:
            print(f"特征保存失败：{str(e)}")
            print(f"错误详情：{str(e.__class__.__name__)}")
            import traceback
            traceback.print_exc()
            return False

    def get_all_users(self):
        if os.path.exists("features.csv"):
            df = pd.read_csv("features.csv")
            return [
                {'user_id': row['user_id'], 'user_name': row['user_name']}
                for _, row in df.iterrows()
            ]
        return []

    def get_attendance_logs(self):
        if os.path.exists(self.log_path):
            try:
                df = pd.read_csv(self.log_path)
                # 确保按时间戳降序排序
                df = df.sort_values('timestamp', ascending=False)
                return df.to_dict('records')
            except Exception as e:
                print(f"读取考勤记录出错：{str(e)}")
                return []
        return []

    def delete_user(self, user_id):
        try:
            print(f"尝试删除用户：{user_id}")
            success = False
            
            # 1. 删除特征数据
            if os.path.exists("features.csv"):
                try:
                    df = pd.read_csv("features.csv")
                    # 确保user_id的类型一致性
                    df['user_id'] = df['user_id'].astype(str)
                    user_id = str(user_id)
                    
                    print(f"当前用户列表：{df['user_id'].tolist()}")
                    print(f"用户信息：\n{df[['user_id', 'user_name']]}")  # 打印用户ID和姓名
                    
                    if user_id in df["user_id"].values:
                        # 在删除前打印要删除的用户信息
                        user_info = df[df['user_id'] == user_id].iloc[0]
                        print(f"正在删除用户：ID={user_id}, 姓名={user_info['user_name']}")
                        
                        df = df[df["user_id"] != user_id]
                        df.to_csv("features.csv", index=False)
                        print(f"已从features.csv中删除用户 {user_id}")
                        success = True
                    else:
                        print(f"用户 {user_id} 不存在于特征数据库中")
                        print(f"数据库中的ID类型：{df['user_id'].dtype}")
                        print(f"要删除的ID类型：{type(user_id)}")
                except Exception as e:
                    print(f"删除特征数据时出错：{str(e)}")
                    return False
            
            # 2. 删除考勤记录
            if os.path.exists(self.log_path):
                try:
                    log_df = pd.read_csv(self.log_path)
                    log_df['user_id'] = log_df['user_id'].astype(str)  # 确保类型一致
                    if user_id in log_df["user_id"].values:
                        log_df = log_df[log_df["user_id"] != user_id]
                        log_df.to_csv(self.log_path, index=False)
                        print(f"已从考勤记录中删除用户 {user_id}")
                        success = True
                except Exception as e:
                    print(f"删除考勤记录时出错：{str(e)}")
            
            # 3. 从内存中删除
            if user_id in self.feature_db:
                user_data = self.feature_db[user_id]
                print(f"从内存中删除用户：ID={user_id}, 姓名={user_data['name']}")
                del self.feature_db[user_id]
                success = True
            
            return success
        except Exception as e:
            print(f"删除用户时出错：{str(e)}")
            return False 