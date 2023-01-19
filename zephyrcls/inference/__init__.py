ZEPHYR_BACKEND_TORCH = 0
ZEPHYR_BACKEND_ONNXRUNTIME = 1
ZEPHYR_BACKEND_MNN = 2


def create_inference(backend: int):
    if backend == ZEPHYR_BACKEND_TORCH:
        from .torch_inference import ClsInferenceTorch
        return ClsInferenceTorch
    elif backend == ZEPHYR_BACKEND_ONNXRUNTIME:
        return NotImplementedError("TODO")
    elif backend == ZEPHYR_BACKEND_MNN:
        return NotImplementedError("TODO")
    else:
        return NotImplementedError("Unknown backend")

