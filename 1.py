import cv2
import numpy as np

# import sys
# print("当前 Python 路径:", sys.executable)
# 生成一张纯黑测试图像
test_image = np.zeros((100, 100, 3), dtype=np.uint8)

# 显示图像（测试 GUI 功能）
cv2.imshow("Test Window", test_image)
cv2.waitKey(1000)  # 显示 1 秒后关闭
cv2.destroyAllWindows()