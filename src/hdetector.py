from utils import *


def draw_boxes(image, boxes, class_ids):
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        color = (0, 255, 0)  # Green color for bounding boxes
        # label = f"Class {class_ids[i]}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        # cv2.putText(image, label, (x_min, y_min - 5), 
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def transpose(boxes, M):
    res = []
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        # [_x_min, _y_min] = np.matmul(np.array([x_min, y_min]), M)
        # [_x_max, _y_max] = np.matmul(np.array([x_max, y_max]), M)
        [_x_min, _y_min] = dot(M, np.array([x_min, y_min]))
        [_x_max, _y_max] = dot(M, np.array([x_max, y_max]))
        res.append(np.array([_x_min, _y_min, _x_max, _y_max]))
    return np.array(res)


def auto_transpose(image, th_dirs, boxes, class_ids, output_dir):
    th0_path = f"{str(Path(th_dirs[0]) / image.replace('rgb8', 'thermal0'))}.png"
    th1_path = f"{str(Path(th_dirs[1]) / image.replace('rgb8', 'thermal1'))}.png"

    th0_boxes = transpose(boxes, rgb_th0)
    th1_boxes = transpose(boxes, rgb_th1)

    # print(th0_boxes)

    """th0_image = draw_boxes(cv2.imread(th0_path), th0_boxes, class_ids)
    th1_image = draw_boxes(cv2.imread(th1_path), th1_boxes, class_ids)"""

    th0_image = draw_boxes(resize_image(cv2.imread(th0_path), SIZE), th0_boxes, class_ids)
    th1_image = draw_boxes(resize_image(cv2.imread(th1_path), SIZE), th1_boxes, class_ids)

    cv2.imwrite(
        f"{str(Path(output_dir) / image.replace('rgb8', 'thermal0'))}.png", th0_image
    )
    cv2.imwrite(
        f"{str(Path(output_dir) / image.replace('rgb8', 'thermal1'))}.png", th1_image
    )


def auto_annotate(
    data,
    det_model="yolov8x.pt",
    sam_model="sam_b.pt",
    device="",
    output_dir=None,
    person_only=False,
):
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()
        index_to_take = range(len(class_ids))
        if person_only:
            index_to_take = [x for x in range(len(class_ids)) if class_ids[x] == 0]
            class_ids = [class_ids[_] for _ in index_to_take]

        if len(class_ids):
            boxes = result.boxes.xyxy[index_to_take]
            auto_transpose(str(Path(result.path).stem), th_dirs, boxes, class_ids, "annotation/auto_th")
            image = cv2.imread(Path(result.path))
            image_with_boxes = draw_boxes(image, boxes, class_ids)
            cv2.imwrite(
                f"{str(Path(output_dir) / Path(result.path).stem)}.png",
                image_with_boxes,
            )


if __name__ == "__main__":
    auto_annotate(
        "D:\IDU5\HumanFinder2000\data\images_rgb",  
        det_model="models\yolov8x.pt",
        sam_model="models\sam_b.pt",
        device="",
        output_dir="annotation/auto_rgb",
        person_only=True,
    )
