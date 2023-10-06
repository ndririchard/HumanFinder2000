from humanFinder2000_utilities import load_images
from ultralytics import YOLO

images = load_images("data/images_rgb")

if __name__ == "__man__":
    
    model = YOLO("model.pt")
    results = model.predict(source=images)
    print(results)