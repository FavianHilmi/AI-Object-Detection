import os
import kagglehub
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()


def main():
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_6f50037781d30c5200c529ca93212a82"
    dataset_path = kagglehub.dataset_download("kapturovalexander/fruits-by-yolo-fruits-detection")
    real_data_path = os.path.join(dataset_path, "Fruits by YOLO", "Fruits by YOLO")

    MODEL_PATH = r"runs\detect\fruits_yolo_model-15\weights\best.pt"

    IMAGE_PATH = os.path.join(real_data_path, "test", "images", "apple_1.jpg") 
    
    if not os.path.exists(IMAGE_PATH):
        test_images_dir = os.path.join(real_data_path, "test", "images")
        first_img = os.listdir(test_images_dir)[0]
        IMAGE_PATH = os.path.join(test_images_dir, first_img)

    print(f"Menggunakan Model : {MODEL_PATH}")
    print(f"Memproses Gambar  : {IMAGE_PATH}")

    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    annotated_frame = results[0].plot()

    window_name = "Object Detection Preview - Fruits"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated_frame)

    print("\n[INFO] Pop-up window terbuka. Klik (X) untuk menutup window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()