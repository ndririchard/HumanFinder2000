from src.annotator import auto_annotate
    
if __name__ == '__main__':
    
    auto_annotate(
        data='/home/geecker/Documents/IDU5/richard-backup/HumanFinder2000/data/images_rgb/', 
        det_model='models/yolov8n.pt', 
        sam_model='models/mobile_sam.pt',
        output_dir='annotation/')