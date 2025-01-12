"""Get availables backends"""

import onnxruntime as onx

print("Onnx runtime version", onx.get_version_string())
print("Avilables backends are :")
for p in onx.get_available_providers():
    print("-", p)
