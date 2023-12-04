import argparse
import os, sys
import tensorrt as trt
import torch_tensorrt

# tensorrt import should be before torch import, otherwise exported model fails during inference
# https://github.com/NVIDIA/TensorRT/issues/1945#issuecomment-1108325943
import torch
import numpy as np
from lib.models import get_net
# from lib.dataset import LoadImages
import onnx
import onnxsim
import onnxruntime as ort
import time


names = ["person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow", "tl_none", "traffic sign", "train"]
colors = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 153, 204], [255, 255, 0], [255, 0, 255], [150, 75, 0], [230, 230, 250], [75, 0, 130], [139, 0, 139], [255, 102, 204], [102, 255, 153]]
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--img_size', type=int, default=640)  # height
    parser.add_argument('--checkpoint', type=str, default='./weights/End-to-end.pth')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', choices=["cuda", "cpu"], type=str, default="cuda")
    parser.add_argument('--mode', choices=["torchscript", "onnx", "tensorrt", "onnx-tensorrt"], type=str, default="onnx-tensorrt")  # onnx-tensorrt
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    if args.device.lower() == "cuda":
        assert torch.cuda.is_available(), "No CUDA device found"

    device = torch.device(args.device)
    print("Device:", device)
    half = args.device.lower() == "cuda"
    
    checkpoint_path = args.checkpoint
    ckpt_dir, ckpt_fname = os.path.split(checkpoint_path)
    model = get_net(None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)  # init img
    
    if half:
        model.half()
        img = img.half()
        print("Model halved on", next(model.parameters()).device)

    model.eval()

    with torch.no_grad():
        for _ in range(10):
            _ = model(img)
    print("warmup done ...")

    # for param in model.parameters():
    #     # print(param.device)
    #     if param.device != device:
    #         print(param)
    
    # dataset = LoadImages(args.source, img_size=args.img_size)
    # img_np = np.zeros((1, 3, args.img_size, args.img_size))
    # print(next(model.parameters()).device)
    # print(img.device)
    
    if args.mode == "onnx":
        onnx_path = export_onnx(model, img, ckpt_dir, ckpt_fname, args)
    elif args.mode == "torchscript":
        ts_model = export_torchscript(model, img, ckpt_dir, ckpt_fname, args)
    elif args.mode == "tensorrt":
        trt_model = export_torch_tensorrt(model, img, ckpt_dir, ckpt_fname, args)
    elif args.mode == "onnx-tensorrt":
        onnx_trt_model = export_onnx_tensorrt(model, img, ckpt_dir, ckpt_fname, args)


def export_onnx_tensorrt(model, img, ckpt_dir, ckpt_fname, args):
    onnx_path = export_onnx(model, img, ckpt_dir, ckpt_fname, args)
    # initialize TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    batch_size_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(batch_size_flag)
    
    # parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(config.get_flag(trt.BuilderFlag.FP16))    # verify flag
        print("**** Using FP16 ****")
        
    time.sleep(10)  # to be able to see above message
    
    print('Building an engine...')
    engine = builder.build_serialized_network(network, config)
    engine_path = os.path.join(ckpt_dir, ckpt_fname.split(".")[0] + "-" + args.device.lower() + ".engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    

# Not working - TODO
def export_torch_tensorrt(model, img, ckpt_dir, ckpt_fname, args):
    # ts_model = export_torchscript(model, img, ckpt_dir, ckpt_fname, args)
    # input_signature = torch_tensorrt.Input(shape=[640, 640], dtype=torch.half),
    # (torch_tensorrt.Input(shape=[64, 64], dtype=torch.half), torch_tensorrt.Input(shape=[64, 64], dtype=torch.half)),)
    trt_ts_module = torch_tensorrt.compile(model, enabled_precisions={torch.half})
    trt_path = os.path.join(ckpt_dir, ckpt_fname.split(".")[0] + "-" + args.device.lower() + ".trt")
    torch.jit.save(trt_ts_module, trt_path)
    return trt_ts_module
    

def export_torchscript(model, img, ckpt_dir, ckpt_fname, args):
    # works for cpu and cuda
    with torch.no_grad():
        traced_model = torch.jit.trace(model, img)
        print(traced_model)
    
    ts_path = os.path.join(ckpt_dir, ckpt_fname.split(".")[0] + "-" + args.device.lower() + ".ts")
    torch.jit.save(traced_model, ts_path)
    return traced_model


# ONNX - works for cpu only; cuda is not working - TODO
def export_onnx(model, img, ckpt_dir, ckpt_fname, args):
    onnx_path = os.path.join(ckpt_dir, ckpt_fname.split(".")[0] + "-" + args.device.lower() + ".onnx")
    with torch.no_grad():
        torch.onnx.export(model, img, onnx_path, export_params=True, input_names=['images'], verbose=True,
                            output_names=['det_out', 'drive_area_seg', 'lane_line_seg'],
                            opset_version=12, 
                            # dynamic_axes={'images':[0], 'det_out':[0], 'drive_area_seg':[0], 'lane_line_seg':[0]}
                            )
        
    print('Model converted to ONNX. Saved at', onnx_path)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = onnxsim.simplify(onnx_model, check_n=3)
    assert check, 'assert check failed'
    onnx.save(onnx_model, onnx_path)
    
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    for ii in sess.get_inputs():
        print("Input: ", ii)
    for oo in sess.get_outputs():
        print("Output: ", oo)
    return onnx_path
    

if __name__ == "__main__":
    main()
    