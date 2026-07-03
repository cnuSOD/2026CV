from ultralytics import YOLO

def main():
    model = YOLO("yolo11-deeper.yaml")

    model.train(data='D:/AASchool/Computer Sience/ComputerVision/AAMAIN/datasets/VisDrone2019/VisDrone2019.yaml',
                epochs=300,
                imgsz=640,
                batch=8,
                device='cpu',
                # augment=False,
                # rect= False
            )

if __name__ == '__main__':
    main()