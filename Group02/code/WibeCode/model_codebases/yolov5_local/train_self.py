from ultralytics import YOLO

def main():
    model = YOLO("yolov5n.yaml")

    model.train(data='D:/Programs/yolo11-self-3/ultralytics-main/datasets/SSDD0.5N/SSDD05N.yaml',
                epochs=300,
                imgsz=640,
                batch=8,
                device='cpu',
                # augment=False,
                # rect= False
            )

if __name__ == '__main__':
    main()