from yolo_utils import initialize_model
import os

def batch_inference(model_path, dataset_path, conf=0.5, save=True):
    model, device = initialize_model(model_path)
    results = model.predict(
        source=dataset_path,
        conf=conf,
        save=save,
        stream=True,
        save_dir = "C:/Users/natha/yoloTestCharm/ourTest_Images"
    )
    for result in results:
        boxes = result.boxes
        for box in boxes:
            print(f"Class: {model.names[int(box.cls.item())]}, "
                  f"Confidence: {box.conf.item():.2f}, "
                  f"Bounding box: {box.xywh.tolist()}")

if __name__ == '__main__':
    dataset_path = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\test"
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"
    batch_inference(model_path, dataset_path)
