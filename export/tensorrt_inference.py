# reference
# https://github.com/NVIDIA/TensorRT/blob/3aaa97b91ee1dd61ea46f78683d9a3438f26192e/samples/python/yolov3_onnx/onnx_to_tensorrt.py#L66
# https://github.com/NVIDIA/TensorRT/blob/3aaa97b91ee1dd61ea46f78683d9a3438f26192e/samples/python/common.py
# https://github.com/NVIDIA/TensorRT/blob/3aaa97b91ee1dd61ea46f78683d9a3438f26192e/samples/python/common.py
# https://github.com/NVIDIA/TensorRT/blob/3aaa97b91ee1dd61ea46f78683d9a3438f26192e/samples/python/yolov3_onnx/data_processing.py


import tensorrt as trt
import numpy as np
import cv2
import ctypes
from cuda import cuda, cudart
import time


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

output_shapes = []

def allocate_buffers(engine: trt.ICudaEngine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding)
        # print(binding, ": ", shape)
        size = trt.volume(shape)
        if engine.has_implicit_batch_dimension:
            size *= engine.max_batch_size
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))

        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
            output_shapes.append(shape)
    return inputs, outputs, bindings, stream


def _do_inference_base(inputs, outputs, stream, execute_async):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    def execute_async():
        context.execute_async_v2(bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async)


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



TRT_LOGGER = trt.Logger()
runtime = trt.Runtime(TRT_LOGGER)
model_path = 'checkpoints/nov14-epoch-100-cpu.engine'

with open(model_path, "rb") as f:
	serialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

inputs, outputs, bindings, stream = allocate_buffers(engine)


def detect():
    img_bgr, img_resized, img = preprocess_image("pictures/0000969_leftImg8bit.jpg")

    inputs[0].host = img
    trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    img_resized_det = img_resized.copy()
    # cv2.imshow("Image", img_resized)

    road = trt_outputs[1][0]
    road = np.argmax(road, axis=0)*255
    road_3d = np.zeros((road.shape[0], road.shape[1], 3), dtype=np.uint8)
    road_3d[:, :, 2] = road
    # cv2.imshow("Road", road_3d)

    ll = trt_outputs[2][0]
    ll = np.argmax(ll, axis=0)*255
    ll_3d = np.zeros((ll.shape[0], ll.shape[1], 3), dtype=np.uint8)
    ll_3d[:, :, 1] = ll
    # cv2.imshow("Lanelines", ll_3d)

    out = cv2.addWeighted(img_resized, 0.7, road_3d, 0.3, 0)
    out = cv2.addWeighted(out, 0.7, ll_3d, 0.3, 0)
    # cv2.imshow("Segmentation", out)

    det_count = {names[i]:0 for i in range(13)}
    for det in trt_outputs[0][0]:
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