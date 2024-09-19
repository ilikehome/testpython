import cv2
import numpy as np
import dlib
import openpose as op

# 加载 OpenPose 模型
params = dict()
params["model_folder"] = "/path/to/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 加载 Dlib 的人脸检测器和形状预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/path/to/shape_predictor_68_face_landmarks.dat")

# 打开视频流（这里以摄像头为例，你也可以指定视频文件路径）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 OpenPose 处理图像
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    op_frame = datum.cvOutputData

    # 将 OpenPose 处理后的图像转换为灰度图用于人脸检测
    gray = cv2.cvtColor(op_frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(gray)

    for face in faces:
        # 使用形状预测器获取人脸特征点
        shape = predictor(gray, face)

        # 提取人脸特征向量（这里只是一个示例，实际可能需要更复杂的处理）
        feature_vector = []
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            feature_vector.append(x)
            feature_vector.append(y)

        # 输出人脸特征向量
        print("Face feature vector:", feature_vector)

    # 显示结果
    cv2.imshow("OpenPose + Face Detection", op_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()