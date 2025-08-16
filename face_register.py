import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

class FaceRegister:
    def __init__(self):
        # 加载模型
        self.detector = YOLO('model/yolov8n-face-lindevs.pt')
        self.session = ort.InferenceSession('model/arcface_w600k_r50.onnx')
        self.feature_db = self.load_features("features.csv")
        self.similarity_threshold = 0.8  # 设置人脸重复检测阈值
    
    def load_features(self, csv_path):
        """加载特征数据库"""
        if not os.path.exists(csv_path):
            print("特征数据库文件不存在！将创建新的数据库")
            return {}
        try:
            df = pd.read_csv(csv_path)
            
            # 显示数据框结构
            print(f"特征文件结构: {df.shape}, 列: {df.columns.tolist()}")
            
            feature_db = {}
            for _, row in df.iterrows():
                try:
                    user_id = str(row["user_id"])
                    user_name = str(row.get("user_name", ""))
                    
                    # 提取特征
                    feature_cols = [col for col in df.columns if col not in ['user_id', 'user_name']]
                    
                    if not feature_cols:
                        print(f"警告：用户 {user_id} 没有特征列")
                        continue
                    
                    features = row[feature_cols].values
                    
                    # 检查特征是否包含非数值并尝试修复
                    try:
                        # 先转换为浮点数
                        features = features.astype(np.float32)
                        
                        # 检查NaN并替换为0
                        if np.isnan(features).any():
                            nan_count = np.isnan(features).sum()
                            print(f"警告：用户 {user_id} 的特征包含 {nan_count} 个NaN值，尝试修复")
                            features = np.nan_to_num(features, nan=0.0)
                    except Exception as e:
                        print(f"警告：用户 {user_id} 的特征无法转换为浮点数: {str(e)}")
                        continue
                    
                    feature_db[user_id] = {
                        'name': user_name,
                        'features': features
                    }
                    print(f"成功加载用户 {user_id} ({user_name}) 的特征，维度: {features.shape}")
                except Exception as e:
                    print(f"加载用户 {user_id} 的特征时出错：{str(e)}")
                    continue
                
            print(f"共加载了 {len(feature_db)} 个用户的特征")
            return feature_db
        except Exception as e:
            print(f"加载特征数据库出错：{str(e)}")
            return {}
    
    def cv2_add_chinese_text(self, img, text, position, text_color=(0, 255, 0), text_size=30):
        """使用PIL绘制中文并转换回OpenCV格式"""
        if isinstance(img, np.ndarray):
            # 将OpenCV图像转换为PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = img
            
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_img)
        # 尝试加载中文字体
        fontpath = "/System/Library/Fonts/PingFang.ttc"  # MacOS中文字体路径
        if not os.path.exists(fontpath):
            # 尝试其他常见字体路径
            alt_fonts = [
                "/System/Library/Fonts/STHeiti Light.ttc",  # MacOS备选字体
                "/System/Library/Fonts/Arial Unicode.ttf",  # 另一备选
                "C:/Windows/Fonts/simhei.ttf",  # Windows中文字体
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"  # Linux中文字体
            ]
            
            for font in alt_fonts:
                if os.path.exists(font):
                    fontpath = font
                    break
        
        try:
            font = ImageFont.truetype(fontpath, text_size)
        except IOError:
            # 无法加载字体，使用默认
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=text_color)
        # 将PIL图像转换回OpenCV格式
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv2_img
    
    def detect_face(self, image_path):
        """从图像中检测人脸"""
        if not os.path.exists(image_path):
            print(f"错误：图片 {image_path} 不存在")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图片 {image_path}")
            return None
            
        results = self.detector(image, verbose=False)
        
        # 如果检测到多个人脸，选择最大的一个
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            
            if len(boxes) > 1:
                print(f"警告：检测到多个人脸，将使用最大的一个")
                # 计算每个人脸框的面积
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                # 选择面积最大的人脸
                largest_face_idx = np.argmax(areas)
                box = boxes[largest_face_idx]
            else:
                box = boxes[0]
                
            x1, y1, x2, y2 = box
            face_img = image[y1:y2, x1:x2]
            
            # 可视化检测结果
            vis_img = image.copy()
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 使用自定义函数添加中文文本
            vis_img = self.cv2_add_chinese_text(vis_img, "检测到的人脸", (x1, y1-35))
            
            cv2.imshow("检测到的人脸", vis_img)
            print("按 q 键关闭窗口...")
            
            # 等待用户按q键
            while True:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                    
            cv2.destroyAllWindows()
            
            return face_img
        else:
            print(f"错误：未在图片中检测到人脸")
            return None
    
    def extract_features(self, face_img):
        """从人脸图像中提取特征"""
        face_img = cv2.resize(face_img, (112, 112))
        face_img = face_img.transpose(2, 0, 1)
        face_img = (face_img / 255.0 - 0.5) / 0.5  # 归一化
        face_img = face_img.astype(np.float32)
        inputs = {self.session.get_inputs()[0].name: np.expand_dims(face_img, 0)}
        features = self.session.run(None, inputs)[0]
        features = features.flatten()
        
        # 检查特征是否包含NaN，如果有则替换为0
        if np.isnan(features).any():
            nan_count = np.isnan(features).sum()
            print(f"警告：提取的特征包含 {nan_count} 个NaN值，将替换为0")
            features = np.nan_to_num(features, nan=0.0)
            
        return features
    
    def check_duplicate(self, features):
        """检查是否有重复人脸"""
        if not self.feature_db:
            return False, None, 0
            
        best_match = None
        best_similarity = 0
        best_name = ""
        
        for uid, user_data in self.feature_db.items():
            try:
                db_feature = user_data['features']
                
                # 确保特征向量不包含NaN
                db_feature = np.nan_to_num(db_feature, nan=0.0)
                
                # 如果维度不匹配，跳过
                if db_feature.size != features.size:
                    print(f"用户 {uid} 的特征维度不匹配: {db_feature.size} vs {features.size}")
                    continue
                
                similarity = cosine_similarity([features], [db_feature])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = uid
                    best_name = user_data['name']
            except Exception as e:
                print(f"比对失败 {uid}: {str(e)}")
                continue
        
        # 如果相似度超过阈值，认为是重复人脸
        if best_similarity > self.similarity_threshold:
            return True, best_match, best_similarity
            
        return False, None, best_similarity
    
    def save_features(self, user_id, user_name, features):
        """保存特征到CSV文件"""
        try:
            print(f"开始保存用户特征：{user_id}, {user_name}")
            
            # 确保特征向量不包含NaN
            if np.isnan(features).any():
                nan_count = np.isnan(features).sum()
                print(f"警告：特征向量包含 {nan_count} 个NaN值，将替换为0")
                features = np.nan_to_num(features, nan=0.0)
            
            features = features.astype(np.float32)
            print(f"特征向量维度: {features.shape}, 数据类型: {features.dtype}")
            
            # 创建新用户的数据行
            feature_list = features.tolist()  # 转换特征为列表
            data = {'user_id': [user_id], 'user_name': [user_name]}
            
            # 为每个特征值创建一个列
            for i, value in enumerate(feature_list):
                data[f'feature_{i}'] = [value]
            
            # 创建新用户的DataFrame
            new_user_df = pd.DataFrame(data)
            
            # 处理已有数据
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
                        # 添加新用户 - 修复pandas警告
                        new_data = new_user_df.copy()
                        # 确保新数据框列与现有数据框匹配
                        for col in existing_df.columns:
                            if col not in new_data.columns:
                                new_data[col] = pd.NA
                        # 只保留现有数据框中的列
                        new_data = new_data[existing_df.columns]
                        # 现在安全地连接两个数据框
                        existing_df = pd.concat([existing_df, new_data], ignore_index=True)
                        print(f"添加新用户 {user_id}")
                    
                    # 保存更新后的数据
                    existing_df.to_csv("features.csv", index=False)
                except Exception as e:
                    print(f"更新特征文件时出错：{str(e)}")
                    print("尝试直接保存新用户数据...")
                    new_user_df.to_csv("features.csv", index=False)
                    return True
            else:
                # 如果文件不存在，直接保存新文件
                new_user_df.to_csv("features.csv", index=False)
                print("创建新的特征文件")
            
            # 更新内存中的特征库
            self.feature_db[str(user_id)] = {
                'name': user_name,
                'features': features
            }
            
            print(f"特征保存成功：{user_id}")
            return True
            
        except Exception as e:
            print(f"特征保存失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def register_face(self, user_id, user_name, image_path):
        """注册人脸"""
        print(f"开始注册用户：学号={user_id}, 姓名={user_name}, 图片={image_path}")
        
        # 检测人脸
        face_img = self.detect_face(image_path)
        if face_img is None:
            return False
        
        # 提取特征
        try:
            features = self.extract_features(face_img)
            print(f"特征提取成功，特征维度：{features.shape}")
        except Exception as e:
            print(f"特征提取失败：{str(e)}")
            return False
        
        # 检查是否有重复人脸
        is_duplicate, dup_id, similarity = self.check_duplicate(features)
        if is_duplicate:
            dup_name = self.feature_db[dup_id]['name']
            print(f"警告：检测到重复人脸！相似度 {similarity:.2f}，与用户 {dup_id} ({dup_name}) 重复")
            print(f"不允许重复录入，注册失败！")
            return False
        else:
            if similarity > 0:
                print(f"人脸相似度检查通过，最高相似度: {similarity:.2f}，低于阈值 {self.similarity_threshold}")
        
        # 保存特征
        return self.save_features(user_id, user_name, features)

def main():
    # 这里可以注册人脸进行识别
    # 在此直接设置参数，无需命令行传递
    user_id = "2001"       # 请修改为实际学号
    user_name = "小明"    # 请修改为实际姓名
    image_path = "/Users/he/Desktop/face.png"  # 请修改为实际图片路径
    
    register = FaceRegister()
    if register.register_face(user_id, user_name, image_path):
        print("注册成功！")
    else:
        print("注册失败！")

if __name__ == "__main__":
    main() 