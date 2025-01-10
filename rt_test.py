import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)


builder = trt.Builder(logger)


def load_engine(file="yolo.engine"):
    with open(file, "rb") as f:
        serialized_engine = f.read()
    return serialized_engine

serialized_engine = load_engine()

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)

context = engine.create_execution_context()

print(context)