from openvino.runtime import Core
import cv2
import numpy as np

# 请手动配置推理计算设备，IR文件路径，图片路径，阈值和标签
DEVICE = "CPU"
ONNX_FileXML = r"C:\Users\NUC\yolov5\yolov5s.onnx"
IMAGE_FILE = r"C:\Users\NUC\yolov5\2.jpeg"
CONF_THRESHOLD = 0.5  #取值0~1
#标签输入
with open(r'C:\Users\NUC\yolov5\coco-labels-2014_2017.txt', 'r') as f:
    LABELS = [x.strip() for x in f]
# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
# 调色板
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# --------------------------- 1. 创建Core对象 --------------------------------------
print("1.Create Core Object.")
ie = Core()

# --------------------------- 2. 载入模型到AI推理计算设备----------------------------
print("2.Load model into device...")
exec_net = ie.compile_model(model=ONNX_FileXML, device_name=DEVICE)

# --------------------------- 3. 准备输入数据 --------------------------------------
# 由OpenCV完成数据预处理：RB交换、Resize，归一化和HWC->NCHW
print("3.Prepare the input data for the model...")
frame = cv2.imread(IMAGE_FILE)
if frame is None:
    raise Exception("Can not read image file: {} by cv2.imread".format(IMAGE_FILE))
# 按照YOLOv5要求，先将图像长:宽 = 1:1，多余部分填充黑边,将图像按最大边1:1放缩
row, col, _ = frame.shape
_max = max(col, row)
im = np.zeros((_max, _max, 3), np.uint8)
im[0:row, 0:col] = frame
blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

# --------------------------- 4. 执行推理并获得结果 ------------------------------------
print("4.Start Inference......")
import time
start = time.time()
result = exec_net.create_infer_request().infer({"images":blob})
for object,value in result.items():
    if object.names == {'output'}:
        outputs = value
end = time.time()

# --------------------------- 5. 处理推理计算结果 --------------------------------------
print("5.Postprocess the inference result......")
# yolov5导出模型输出为：
# output层形状为[1,N,85], 其中N为预测框的个数，85为[cx, cy, w, h, score, 80个类的得分]
# 找出output层 且 score >= CONF_THRESHOLD的结果
class_ids = []
confidences = []
boxes = []
output_data = outputs[0]
print(output_data)
rows = output_data.shape[0]

image_width, image_height, _ = im.shape
x_factor = image_width / INPUT_WIDTH
y_factor = image_height / INPUT_HEIGHT

for r in range(rows):
    row = output_data[r]
    confidence = row[4]
    if confidence >= 0.4:
        classes_scores = row[5:]
        _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
        class_id = max_indx[1]
        if (classes_scores[class_id] > .25):
            confidences.append(float(confidence))
            class_ids.append(class_id)
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

result_class_ids = []
result_confidences = []
result_boxes = []

for i in indexes:
    result_class_ids.append(np.array(class_ids)[i])
    result_boxes.append(np.array(boxes)[i])
    result_confidences.append(np.array(confidences)[i])

# 显示检测框bbox
for (classid, confidence, box) in zip(result_class_ids, result_confidences, result_boxes): 
    color = colors[int(classid) % len(colors)]
    print(box)
    print(color)
    cv2.rectangle(frame, box, color, 2)
    cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
    cv2.putText(frame, LABELS[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

cv2.imshow("Detection results", frame)
cv2.waitKey()
cv2.destroyAllWindows()
print("All is completed!")
