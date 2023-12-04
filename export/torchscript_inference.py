import torch
import numpy as np
import cv2
import time


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
names = ["person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow", "tl_none", "traffic sign", "train"]

def preprocess_image(img_path):
    img_bgr = cv2.imread(img_path)
    img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_raw, (640, 640))
    img = img_resized/255.0
    img = (img-mean)/std
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32, order="C")
    return img_bgr, img_resized, img



model_path = 'checkpoints/nov14-epoch-100.ts'

model = torch.jit.load(model_path)

def detect():
    img_bgr, img_resized, img = preprocess_image("pictures/0000969_leftImg8bit.jpg")

    with torch.no_grad():
        trt_outputs = model(torch.from_numpy(img).cuda().half())

    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    img_resized_det = img_resized.copy()
    # cv2.imshow("Image", img_resized)

    road = trt_outputs[0][0].cpu().numpy()
    road = np.argmax(road, axis=0)*255
    road_3d = np.zeros((road.shape[0], road.shape[1], 3), dtype=np.uint8)
    road_3d[:, :, 2] = road
    # cv2.imshow("Road", road_3d)

    ll = trt_outputs[1][0].cpu().numpy()
    ll = np.argmax(ll, axis=0)*255
    ll_3d = np.zeros((ll.shape[0], ll.shape[1], 3), dtype=np.uint8)
    ll_3d[:, :, 1] = ll
    # cv2.imshow("Lanelines", ll_3d)

    out = cv2.addWeighted(img_resized, 0.7, road_3d, 0.3, 0)
    out = cv2.addWeighted(out, 0.7, ll_3d, 0.3, 0)
    # cv2.imshow("Segmentation", out)

    det_count = {names[i]:0 for i in range(13)}

    for det in trt_outputs[2][0].cpu().numpy():
        if det[4]> 0.5:
            cx = det[0]
            cy = det[1]
            w = det[2]
            h = det[3]
            x1, x2 = cx-w/2, cx+w/2
            y1, y2 = cy-h/2, cy+h/2
            class_id = np.argmax(det[5:])
            det_count[names[class_id]] += 1
            cv2.rectangle(img_resized_det, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # cv2.imshow("Detections", img_resized_det)
    # print(det_count)
    # cv2.waitKey(0)



# warmup
for _ in range(3):
    detect()

count = 1000
start = time.perf_counter()
for _ in range(count):
    detect()
end = time.perf_counter()
print("Duration:{} ms".format((end-start)/count))