import torch


class DeviceResolver:
    @staticmethod
    def resolve_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
