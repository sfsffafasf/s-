import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFont

# 启用详细日志
DEBUG = True

# 定义一组鲜艳的颜色 (BGR格式)
VIBRANT_COLORS = [
    (0, 255, 255),   # 黄色
    (255, 0, 255),   # 紫色
    (0, 255, 0),     # 绿色
    (255, 0, 0),     # 蓝色
    (0, 0, 255),     # 红色
    (0, 165, 255),   # 橙色
    (255, 191, 0),   # 深天蓝
    (255, 20, 147),  # 深粉色
    (127, 0, 255),   # 紫罗兰
    (255, 0, 127),   # 玫瑰红
    (127, 255, 0),   # 草绿
    (0, 140, 255),   # 棕色
    (0, 215, 255),   # 金色
    (128, 128, 240)  # 浅紫
]

class FaceDetectionRecognition:
    def __init__(self, mode=1):
        self.mode = mode
        
        if mode == 1:
            # 模式1: 人脸检测与识别
            print("模式1: 人脸检测与识别")
            print("正在加载YOLO人脸检测模型...")
            self.detector = YOLO('model/yolov8n-face-lindevs.pt')
            print("正在加载ArcFace模型...")
            self.session = ort.InferenceSession('model/arcface_w600k_r50.onnx')
            print("正在加载特征库...")
            self.feature_db = self.load_features("features.csv")
            self.threshold = 0.4  # 设置比对阈值
            print(f"特征比对阈值设置为: {self.threshold}")
        else:
            # 模式2: 通用目标检测
            print("模式2: 通用目标检测")
            print("正在加载YOLO通用目标检测模型...")
            try:
                self.detector = YOLO('yolov8n.pt')  # 使用标准YOLOv8模型
                print("模型加载完成")
            except Exception as e:
                print(f"加载模型出错: {str(e)}")
                print("尝试从本地目录加载模型...")
                try:
                    self.detector = YOLO('model/yolov8n.pt')  # 尝试从model目录加载
                    print("从model目录成功加载模型")
                except Exception as e2:
                    print(f"加载本地模型失败: {str(e2)}")
                    print("请确保模型文件存在")
                    raise
            
            # 为每个类别预生成一个固定的颜色
            self.class_colors = {}
            
    def get_color_for_class(self, class_id):
        """为类别ID获取一个固定的鲜艳颜色"""
        if class_id not in self.class_colors:
            # 如果类别ID在预定义颜色范围内，使用预定义颜色
            if class_id < len(VIBRANT_COLORS):
                self.class_colors[class_id] = VIBRANT_COLORS[class_id]
            else:
                # 否则生成一个随机的鲜艳颜色
                h = (class_id * 33) % 360   # 色相 (0-359)
                s = 100                     # 饱和度 100%
                v = 100                     # 亮度 100%
                
                # 将HSV转换为RGB
                h_i = int(h / 60)
                f = h / 60 - h_i
                p = v * (1 - s/100)
                q = v * (1 - f * s/100)
                t = v * (1 - (1 - f) * s/100)
                
                if h_i == 0:
                    r, g, b = v, t, p
                elif h_i == 1:
                    r, g, b = q, v, p
                elif h_i == 2:
                    r, g, b = p, v, t
                elif h_i == 3:
                    r, g, b = p, q, v
                elif h_i == 4:
                    r, g, b = t, p, v
                else:
                    r, g, b = v, p, q
                
                # 将0-100的值映射到0-255
                r = int(r * 255 / 100)
                g = int(g * 255 / 100)
                b = int(b * 255 / 100)
                
                # 存储为BGR格式
                self.class_colors[class_id] = (b, g, r)
                
        return self.class_colors[class_id]

    def load_features(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"错误：特征数据库文件 {csv_path} 不存在！")
            return {}
        try:
            print(f"尝试读取特征文件: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 显示数据框结构
            print(f"特征文件结构: {df.shape}, 列: {df.columns.tolist()}")
            
            feature_db = {}
            for _, row in df.iterrows():
                try:
                    user_id = str(row["user_id"])
                    user_name = str(row.get("user_name", ""))
                    
                    # 提取特征，排除user_id和user_name列
                    feature_cols = [col for col in df.columns if col not in ['user_id', 'user_name']]
                    
                    if not feature_cols:
                        print(f"警告：用户 {user_id} 没有特征列")
                        continue
                    
                    features = row[feature_cols].values
                    
                    # 检查特征是否包含非数值并修复
                    try:
                        features = features.astype(np.float32)
                        
                        # 检查NaN并替换为0，而不是跳过用户
                        if np.isnan(features).any():
                            nan_count = np.isnan(features).sum()
                            print(f"警告：用户 {user_id} 的特征包含 {nan_count} 个NaN值，将替换为0")
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
                    print(f"错误详情: {type(e).__name__}: {str(e)}")
                    continue
                
            print(f"共加载了 {len(feature_db)} 个用户的特征")
            if len(feature_db) == 0:
                print("警告：没有加载到任何用户的特征，识别功能将无法正常工作！")
                print("请先使用 face_register.py 注册人脸")
            return feature_db
        except Exception as e:
            print(f"加载特征数据库出错：{str(e)}")
            print(f"错误详情: {type(e).__name__}: {str(e)}")
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

    def extract_features(self, face_img):
        if DEBUG:
            print(f"提取人脸特征，输入图像大小: {face_img.shape}")
            
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
            
        if DEBUG:
            print(f"特征提取完成，特征维度: {features.shape}")
            
        return features

    def compare_face(self, current_feature):
        if not self.feature_db:
            if DEBUG:
                print("特征库为空，无法进行比对")
            return None, "", 0
            
        best_match = None
        best_similarity = 0
        best_name = ""
        
        if DEBUG:
            print(f"开始比对，特征库中有 {len(self.feature_db)} 个用户")
            
        for uid, user_data in self.feature_db.items():
            try:
                db_feature = user_data['features']
                
                # 确保特征向量不包含NaN
                if np.isnan(db_feature).any():
                    db_feature = np.nan_to_num(db_feature, nan=0.0)
                
                # 安全检查
                if not isinstance(db_feature, np.ndarray):
                    print(f"用户 {uid} 的特征不是numpy数组，跳过")
                    continue
                    
                if db_feature.size != current_feature.size:
                    print(f"用户 {uid} 的特征维度不匹配: {db_feature.size} vs {current_feature.size}")
                    continue
                
                # 计算相似度
                similarity = cosine_similarity([current_feature], [db_feature])[0][0]
                
                if DEBUG:
                    print(f"用户 {uid} ({user_data['name']}) 的相似度: {similarity:.4f}")
                    
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = uid
                    best_name = user_data['name']
            except Exception as e:
                print(f"比对失败 {uid}: {str(e)}")
                continue

        if best_match and best_similarity > self.threshold:
            if DEBUG:
                print(f"找到最佳匹配: {best_match} ({best_name}), 相似度: {best_similarity:.4f}")
            return best_match, best_name, best_similarity
        else:
            if DEBUG:
                if best_match:
                    print(f"最高相似度 {best_similarity:.4f} 低于阈值 {self.threshold}，未能识别")
                else:
                    print(f"未找到匹配的人脸")
            return None, "", 0

    def run_face_detection(self):
        """模式1：人脸检测与识别"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        print("摄像头已打开，开始人脸检测和识别")
        print(f"按Q键退出程序")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像")
                break
                
            # 使用YOLO检测人脸
            results = self.detector(frame, verbose=False)
            
            # 处理检测结果
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                if DEBUG and len(boxes) > 0:
                    print(f"检测到 {len(boxes)} 个人脸")
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confs[i]
                    
                    # 提取人脸图像
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    # 默认使用鲜艳的橙色框和标签表示未识别的人脸
                    color = (0, 165, 255)  # 橙色 (B,G,R)
                    label = f"人脸: {conf:.2f}"
                    
                    try:
                        # 提取特征并比对
                        current_feature = self.extract_features(face_img)
                        user_id, user_name, similarity = self.compare_face(current_feature)
                        
                        # 如果比对成功，使用绿色框并显示姓名
                        if user_id:
                            color = (0, 255, 0)  # 绿色
                            label = f"{user_name}: {similarity:.2f}"
                            if DEBUG:
                                print(f"识别成功：{user_id} {user_name}, 相似度: {similarity:.4f}")
                    except Exception as e:
                        print(f"特征提取或比对出错：{str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # 绘制人脸框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # 更粗的线条
                    
                    # 在边界框上方绘制一个填充的矩形作为标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
                    
                    # 使用PIL添加中文文本
                    frame = self.cv2_add_chinese_text(frame, label, (x1, y1-35), 
                                                     text_color=(255, 255, 255))  # 白色文字
            
            # 显示结果和帧率
            cv2.putText(frame, f"按Q退出", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("人脸检测与识别", frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def run_object_detection(self):
        """模式2：通用目标检测"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        print("摄像头已打开，开始通用目标检测")
        print(f"按Q键退出程序")
        
        # COCO数据集的类别名称（简体中文）
        class_names = [
            "人", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", "船", 
            "交通灯", "消防栓", "停止标志", "停车计时器", "长凳", "鸟", "猫", "狗", "马", 
            "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", 
            "手提箱", "飞盘", "滑雪板", "滑雪板", "运动球", "风筝", "棒球棒", "棒球手套", 
            "滑板", "冲浪板", "网球拍", "瓶子", "葡萄酒杯", "杯子", "叉子", "刀", "勺子", 
            "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "比萨", 
            "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌", "厕所", "电视", 
            "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", 
            "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像")
                break
                
            # 使用YOLO进行目标检测
            results = self.detector(frame, verbose=False)
            
            # 处理检测结果
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                if DEBUG and len(boxes) > 0:
                    print(f"检测到 {len(boxes)} 个物体")
                
                for i, box in enumerate(boxes):
                    if confs[i] < 0.3:  # 过滤低置信度的检测结果
                        continue
                        
                    x1, y1, x2, y2 = box
                    conf = confs[i]
                    cls_id = cls_ids[i]
                    
                    # 获取类别名称
                    if cls_id < len(class_names):
                        class_name = class_names[cls_id]
                    else:
                        class_name = f"类别{cls_id}"
                    
                    # 为每个类别获取一个固定的鲜艳颜色
                    color = self.get_color_for_class(cls_id)
                    
                    # 标签文字
                    label = f"{class_name}: {conf:.2f}"
                    
                    # 绘制边界框 (更粗的线条)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # 在边界框上方绘制一个填充的矩形作为标签背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
                    
                    # 使用PIL添加中文文本 (白色文字)
                    frame = self.cv2_add_chinese_text(frame, label, (x1, y1-35), 
                                                     text_color=(255, 255, 255))
            
            # 显示结果和帧率
            cv2.putText(frame, f"按Q退出", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("通用目标检测", frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """根据模式运行相应的功能"""
        if self.mode == 1:
            self.run_face_detection()
        else:
            self.run_object_detection()

def parse_args():
    """解析命令行参数"""
    # 模型1是使用的人脸检测的模型，模式2是通用的目标检测模型
    parser = argparse.ArgumentParser(description='人脸检测与通用目标检测程序')
    parser.add_argument('--mode', type=int, default=2, choices=[1, 2],
                       help='运行模式：1=人脸检测与识别，2=通用目标检测')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"初始化系统，运行模式：{args.mode}")
    face_system = FaceDetectionRecognition(mode=args.mode)
    print("开始运行，按Q键退出")
    face_system.run() 