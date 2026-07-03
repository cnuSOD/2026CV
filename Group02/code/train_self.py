from ultralytics import YOLO

def main():
    model = YOLO("uavyolo.yaml")

    model.train(data='VisDrone.yaml',
                epochs=300,
                imgsz=640,
                batch=8,
                device='cuda',
                resume=True,
            )

if __name__ == '__main__':
    main()