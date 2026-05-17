import os
import torch
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


class TensorRTInfer:
    def __init__(self, engine_path):
        if not HAS_TENSORRT:
            raise ImportError("TensorRT or PyCUDA not installed.")
        
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(self.engine.get_tensor_shape(name))
            else:
                self.output_name = name
                self.output_shape = tuple(self.engine.get_tensor_shape(name))
        
        self.input_dtype = np.float32
        self.output_dtype = np.float32
        
        self.d_input = cuda.mem_alloc(np.zeros(self.input_shape, dtype=self.input_dtype).nbytes)
        self.d_output = cuda.mem_alloc(np.zeros(self.output_shape, dtype=self.output_dtype).nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def infer(self, input_data):
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        output = np.empty(self.output_shape, dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output


def export_to_onnx(model, input_shape, onnx_path, device='cuda'):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[+] Exported ONNX to {onnx_path}")


def build_tensorrt_engine(onnx_path, engine_path, fp16=True, max_batch_size=8):
    if not HAS_TENSORRT:
        print("[-] TensorRT not available. Install tensorrt and pycuda.")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    config.max_batch_size = max_batch_size
    
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    profile.set_shape(input_name, (1,) + input_shape[1:], (max_batch_size//2,) + input_shape[1:], (max_batch_size,) + input_shape[1:])
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"[+] TensorRT engine built: {engine_path}")
