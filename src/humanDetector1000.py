from humanFinder2000_utilities import load_images
from ultralytics import YOLO

images = load_images("data/images_rgb")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    results = model.predict(source=images)
    print(results)