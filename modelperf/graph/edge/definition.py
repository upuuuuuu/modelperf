class Edge:
    def __init__(self):
        self.id = ""
        self.src = ""
        self.tgt = ""

        self.delivered_tensor_info = TensorInfo()

    def to_dict_ir(self) -> dict:
        return {
            "id": self.id,
            "src": self.src,
            "tgt": self.tgt,
        }

    def to_dict_cg(self) -> dict:
        return {
            "id": self.id,
            "src": self.src,
            "tgt": self.tgt,
            "delivered_tensor_info": self.delivered_tensor_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        obj = cls()
        obj.id = d.get("id", "")
        obj.src = d.get("src", "")
        obj.tgt = d.get("tgt", "")
        if "delivered_tensor_info" in d:
            obj.delivered_tensor_info = TensorInfo.from_dict(d["delivered_tensor_info"])

        return obj


class TensorInfo:
    def __init__(self):
        self.tensor_shape = []
        self.tensor_dtype = ""
        self.tensor_dtype_size = 0
        self.tensor_bytes = 0

    def to_dict(self) -> dict:
        return {
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "tensor_dtype_size": self.tensor_dtype_size,
            "tensor_bytes": self.tensor_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TensorInfo":
        obj = cls()
        obj.tensor_shape = d.get("tensor_shape", [])
        obj.tensor_dtype = d.get("tensor_dtype", "")
        obj.tensor_dtype_size = d.get("tensor_dtype_size", 0)
        obj.tensor_bytes = d.get("tensor_bytes", 0)
        return obj
