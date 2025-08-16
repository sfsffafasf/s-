from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
import base64
from face_system import FaceRecSystem
from datetime import datetime
import uvicorn
app = FastAPI()

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化人脸识别系统
face_system = FaceRecSystem()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # 获取统计数据
    total_users = len(face_system.get_all_users())
    logs = face_system.get_attendance_logs()
    
    # 计算今日打卡人数（去重）
    today = datetime.now().strftime("%Y-%m-%d")
    today_logs = [log for log in logs if log['timestamp'].startswith(today)]
    today_attendance = len(set(log['user_id'] for log in today_logs))  # 使用set去重
    
    # 总打卡次数
    total_attendance = len(logs)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "total_users": total_users,
        "today_attendance": today_attendance,  # 今日打卡人数（去重）
        "total_attendance": total_attendance   # 总打卡次数
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register_face")
async def register_face(
    user_id: str = Form(...), 
    user_name: str = Form(...), 
    image: UploadFile = File(...)
):
    try:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        bboxes = face_system.detect_faces(img)
        if len(bboxes) == 0:
            return JSONResponse({"status": "error", "message": "未检测到人脸"})
            
        x1, y1, x2, y2 = bboxes[0]
        face_img = img[y1:y2, x1:x2]
        features = face_system.extract_features(face_img)
        
        # 保存特征和用户信息
        face_system.save_features(user_id, user_name, features)
        
        return JSONResponse({"status": "success", "message": "人脸注册成功"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page(request: Request):
    return templates.TemplateResponse("attendance.html", {"request": request})

@app.post("/check_attendance")
async def check_attendance(image_data: str = Form(...)):
    try:
        # 确保图像数据是Base64格式
        if ',' in image_data:
            # 分割header和实际的base64数据
            header, encoded_data = image_data.split(',', 1)
        else:
            encoded_data = image_data

        # 解码Base64图像数据
        try:
            image_bytes = base64.b64decode(encoded_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return JSONResponse({
                    "status": "error",
                    "message": "无法解码图像数据"
                })
                
        except Exception as e:
            print(f"图像解码错误: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"图像解码错误: {str(e)}"
            })

        # 进行人脸识别
        user_id, frame_with_face, user_name = face_system.check_in(img)
        if user_id:
            # 将处理后的图像（带框和姓名）转换为base64
            _, buffer = cv2.imencode('.jpg', frame_with_face)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JSONResponse({
                "status": "success", 
                "message": f"打卡成功：{user_name}",
                "user_id": user_id,
                "frame": frame_base64,  # 返回带标注的图像
            })
        return JSONResponse({
            "status": "error", 
            "message": "未识别到已注册的人脸"
        })
    except Exception as e:
        print(f"打卡处理错误: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })

@app.get("/management", response_class=HTMLResponse)
async def management_page(request: Request):
    users = face_system.get_all_users()
    all_logs = face_system.get_attendance_logs()
    
    # 创建用户ID到姓名的映射
    user_names = {user['user_id']: user['user_name'] for user in users}
    
    # 处理所有考勤记录
    attendance_records = []
    for log in all_logs:
        user_id = log['user_id']
        timestamp = log['timestamp']
        date, time = timestamp.split(' ')
        
        record = {
            'user_id': user_id,
            'user_name': user_names.get(user_id, "未知用户"),
            'date': date,
            'status': True,  # 既然有记录就是已打卡
            'check_time': time
        }
        attendance_records.append(record)
    
    # 按日期和时间降序排序
    attendance_records.sort(key=lambda x: x['date'] + ' ' + x['check_time'], reverse=True)
    
    return templates.TemplateResponse(
        "management.html", 
        {
            "request": request, 
            "users": users, 
            "attendance_records": attendance_records
        }
    )

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    try:
        success = face_system.delete_user(user_id)
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"用户 {user_id} 删除成功"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"用户 {user_id} 删除失败"
            }, status_code=400)
    except Exception as e:
        print(f"删除用户时出错：{str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)