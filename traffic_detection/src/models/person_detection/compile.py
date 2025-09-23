import rebel
import torch
from ultralytics import YOLO


def main():
    model_name = "yolo11n-seg"

    model = YOLO(model_name + ".pt").model
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
