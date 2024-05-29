import cv2
import os
from Class.Positioning import CameraMovementTracker
import json
from Class import ImageSimilarityChecker

tracker = CameraMovementTracker()
user = "NPC-AI"
detected_objects = []


# UAP ve UAI inilebilir kontrolü yapacak olan modelin oluşturulması
image_similarity_checker = ImageSimilarityChecker.ImageSimilarityChecker()
def formatter(results,path,img):
    tracker.process_frame(cv2.imread(os.path.join(path, img)))
    print(tracker.get_positions())

    # Veri yapısı
    data = {
        "id": img.split(".")[0],
        "user": user,
        "frame": img.split(".")[0],
    }

    # Algılanan nesnelerin JSON formatına dönüştürüleceği listeyi oluştur
    detected_objects_json = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            obj = {
                "cls": int(class_id),
                "landing_status": None,
                "top_left_x": x1,
                "top_left_y": y1,
                "bottom_right_x": x2,
                "bottom_right_y": y2
            }
            if class_id == 3 or class_id == 2:
                if image_similarity_checker.control(x1=x1, y1=y1, x2=x2, y2=y2, image_path=os.path.join(path, img),class_id=class_id):

                    obj["landing_status"] = 1
                else:
                    obj["landing_status"] = 0
            else:
                obj["landing_status"] = -1
            detected_objects_json.append(obj)

    # Algılanan çevirilerin JSON formatına dönüştürüleceği listeyi oluştur
    detected_translations_json = []
    translation = tracker.get_positions().tolist()  # Get the current position
    x, y = translation  # Unpack the translation
    detected_translations_json.append({
        "translation_x": x,
        "translation_y": y
    })

    # Veriyi JSON uyumlu hale getir
    json_data = {
        "id": data["id"],
        "user": data["user"],
        "frame": data["frame"],
        "detected_objects": detected_objects_json,
        "detected_translations": detected_translations_json
    }

    # JSON dosyasına yazma işlemi
    json_file_path = f"json/Result_{img.split('.')[0]}.json"  # Dilediğiniz dosya adını ve yolunu belirleyebilirsiniz
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    return tracker.get_positions().tolist()