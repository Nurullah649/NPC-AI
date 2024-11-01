import cv2
import os

from colorama import Fore

from Class.Positioning_for_yolo import CameraMovementTracker
import json
from Class import Does_it_intersect


tracker = CameraMovementTracker()
detected_objects = []

def formatter(results,path,name):
    tracker.process_frame(cv2.imread(path))
    print(tracker.get_positions())
    detected_objects_json = []
    # Algılanan nesnelerin JSON formatına dönüştürüleceği listeyi oluştur
    if results is None:
        detected_objects_json.append(None)
    else:

        for result in results:
            objects=result.boxes.data.tolist()
            for r in objects:
                x1, y1, x2, y2, score, class_id = r
                obj = {
                    "cls": f"{str(int(class_id))}",
                    "landing_status": None,
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2
                }
                if class_id == 3 or class_id == 2:
                    if Does_it_intersect.does_human_center_intersect(results,path):
                        print(Fore.GREEN,'İNİLEBİLİR')
                        obj["landing_status"] = "1"
                    else:
                        print(Fore.RED,'İNİLEMEZ')
                        obj["landing_status"] = "0"
                else:
                    obj["landing_status"] = "-1"
                detected_objects_json.append(obj)

    # Algılanan çevirilerin JSON formatına dönüştürüleceği listeyi oluştur

    translation = tracker.get_positions().tolist()  # Get the current position
    x, y = translation  # Unpack the translation

    detected_translation = [{
        "translation_x": x,
        "translation_y": y
    }
    ]
    json_data = {
        "id": "1",
        "user": "NPC-AI",
        "frame": name,
        "detected_objects": detected_objects_json,
        "detected_translations": detected_translation
    }
    '''if not os.path.exists("json"):
        os.makedirs("json")
    # JSON dosyasına yazma işlemi
    json_file_path = f"json/{name.split('.jpg')[0]}.json"  # Dilediğiniz dosya adını ve yolunu belirleyebilirsiniz
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
        print(f"JSON dosyası oluşturuldu: {json_file_path}")'''
    return x,y