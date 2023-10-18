from src.utils import *

def draw_bounding_boxes(image, boxes=[], color=(0, 0, 255), thickness=2):
    image_with_box = image.copy()
    for box in boxes.tolist():
        start, end = (int(box[0]), int(box[3])) , (int(box[2]), int(box[1]))
        cv2.rectangle(image_with_box, start, end, color, thickness)
    return image_with_box

def auto_annotate(
        data, det_model='yolov8x.pt', 
        sam_model='sam_b.pt', device='', output_dir=None):
    
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)
    data = Path(data)

    if not output_dir:
        output_dir = data.parent / f'{data.stem}_auto_annotate_labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:

        class_ids = result.boxes.cls.int().tolist()

        # get index of 0 (person) in class_ids
        person_index = [x for x in range(len(class_ids)) if class_ids[x] == 0]

        # get boxes of person_index
        boxes = result.boxes.xyxy[person_index]

        # get image
        img = result.orig_img

        # draw boxes
        img_with_boxes = draw_bounding_boxes(img, boxes)

        # save image
        # Create a Path object for the output path
        output_path = Path(output_dir) / f'{Path(result.path).stem}.png'
        
        # Save the image with bounding boxes
        image_pil = Image.fromarray(img_with_boxes)
        image_pil.save(output_path)
    

            

