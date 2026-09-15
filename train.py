import os
import shutil
import yaml
import kagglehub
import pandas as pd
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

def setup_yolo_structure(base_path, split_name):
    """Menata folder dan mengonversi _classes.csv ke format .txt YOLO."""
    target_dir = os.path.join(base_path, split_name)
    csv_path = os.path.join(target_dir, "_classes.csv")
    
    if not os.path.exists(csv_path):
        return

    images_dir = os.path.join(target_dir, "images")
    labels_dir = os.path.join(target_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for item in os.listdir(target_dir):
        if item.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(target_dir, item)
            dst = os.path.join(images_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)

    df = pd.read_csv(csv_path, sep=',', skipinitialspace=True)

    for _, row in df.iterrows():
        img_name = str(row.iloc[0]).strip()
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)

        active_classes = []
        values = row.iloc[1:].values
        for idx, val in enumerate(values):
            try:
                if int(val) == 1:
                    active_classes.append(idx)
            except (ValueError, TypeError):
                continue

        with open(txt_path, "w") as f:
            for cls_id in active_classes:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")
                
def main():
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_6f50037781d30c5200c529ca93212a82"
    
    dataset_path = kagglehub.dataset_download("kapturovalexander/fruits-by-yolo-fruits-detection")
    print("Dataset path:", dataset_path)
    
    real_data_path = os.path.join(dataset_path, "Fruits by YOLO", "Fruits by YOLO")
    print("Mengonversi _classes.csv ke format label YOLO (.txt)...")
    for split in ["train", "valid", "test"]:
        setup_yolo_structure(real_data_path, split)

    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
        
    data_config["path"] = real_data_path
    data_config["train"] = "train"
    data_config["val"] = "valid"
    data_config["test"] = "test"

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f)

    for split in ["train", "valid", "test"]:
        for root, _, files in os.walk(os.path.join(real_data_path, split)):
            for file in files:
                if file.endswith(".cache"):
                    os.remove(os.path.join(root, file))
                    
    print("Memulai Training YOLOv8...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path, 
        epochs=10, 
        imgsz=320,
        device=0,
        batch=16, 
        name="fruits_yolo_model"
    )

if __name__ == "__main__":
    main()