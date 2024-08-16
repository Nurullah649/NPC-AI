import cv2
import os
import predict_with_server
from Class.Positioning_for_Server import CameraMovementTracker
import json
from Class import Does_it_intersect
import re

tracker = CameraMovementTracker(predict_with_server.first_translation_data)
detected_objects = []
BASE_URL = "http://teknofest.cezerirobot.com:1025/"

def extract_number_from_url(url):
    # URL'deki sayıyı aramak için regex deseni
    pattern = re.compile(r'\d+')
    match = pattern.search(url)

    if match:
        return match.group(0)  # Eşleşen sayıyı döndür
    else:
        return None  # Eğer sayı bulunamazsa None döndür
def formatter(results,path,data,name):

    tracker.process_frame(cv2.imread(path))
    print(tracker.get_positions())
    detected_objects_json = []
    # Algılanan nesnelerin JSON formatına dönüştürüleceği listeyi oluştur
    if results is None:
        detected_objects_json.append(None)
    else:
        #Does_it_intersect.does_it_intersect(results)
        for result in results:
            objects=result.boxes.data.tolist()
            for r in objects:
                x1, y1, x2, y2, score, class_id = r
                obj = {
                    "cls": f"{BASE_URL}classes/{str(int(class_id+1))}/",
                    "landing_status": None,
                    "top_left_x": x1,
                    "top_left_y": y1,
                    "bottom_right_x": x2,
                    "bottom_right_y": y2
                }
                if class_id == 3 or class_id == 2:
                    if Does_it_intersect.does_human_center_intersect(result,path):
                        obj["landing_status"] = "1"
                    else:
                        obj["landing_status"] = "0"
                else:
                    obj["landing_status"] = "-1"
                detected_objects_json.append(obj)

    # Algılanan çevirilerin JSON formatına dönüştürüleceği listeyi oluştur
    if data["translation_data"]["health_status"] == "0":
        translation = tracker.get_positions().tolist()  # Get the current position
        x, y = translation  # Unpack the translation
    else:
        x, y = data["translation_data"]["translation_x"], data["translation_data"]["translation_y"]
    detected_translation = [{
        "translation_x": x,
        "translation_y": y
    }
    ]
    json_data = {
        "id": extract_number_from_url(data["frame_data"]["url"]),
        "user": predict_with_server.USER_URL,
        "frame": f"{data['frame_data']['url']}",
        "detected_objects": detected_objects_json,
        "detected_translations": detected_translation
    }
    if not os.path.exists("json"):
        os.makedirs("json")
    # JSON dosyasına yazma işlemi
    json_file_path = f"json/{name.split('.jpg')[0]}.json"  # Dilediğiniz dosya adını ve yolunu belirleyebilirsiniz
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
        print(f"JSON dosyası oluşturuldu: {json_file_path}")
    return json_data