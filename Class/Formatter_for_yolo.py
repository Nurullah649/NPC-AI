import cv2
import os
import numpy as np
from colorama import Fore
from sklearn.linear_model import LinearRegression
from .Calculate_Direction import Calculate_Direction
from .DPVO_Obejct import DPVO_object
import json
from Class import Does_it_intersect

calibration_frames = []
gt_data = []
positions_data=[]
detected_objects = []
scale_factor = None
detected=None
offset=None
""" ORB için kullanılan kalibrasyon dosyası
def read_calibration_file():
    camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = read_calibration_file()"""
tracker = DPVO_object()

def formatter(results,path,idx,gt_data_,health_status):
    global scale_factor,detected,offset

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
                    if Does_it_intersect.does_other_center_intersect(results,path):
                        print(Fore.GREEN,'İNİLEBİLİR')
                        obj["landing_status"] = "1"
                    else:
                        print(Fore.RED,'İNİLEMEZ')
                        obj["landing_status"] = "0"
                else:
                    obj["landing_status"] = "-1"
                detected_objects_json.append(obj)

    # Algılanan çevirilerin JSON formatına dönüştürüleceği listeyi oluştur

    translation = tracker.process_frames_from_list(idx,path)  # Get the current position
    print(translation)
    x, y ,z= translation  # Unpack the translation
    if health_status == '1':
        calibration_frames.append((x, y))
        gt_data.append([float(gt_data_[0]), float(gt_data_[1])])
        positions_data.append([x, y])
        x,y=gt_data_[0], gt_data_[1]
    elif health_status == '0':
        if scale_factor is None:
            detected = Calculate_Direction(gt_data=gt_data, alg_data=positions_data)
            if  detected.calculate_direction_change():
                gt_positions = np.array(gt_data)
                alg_positions = np.array(positions_data)
                model = LinearRegression(fit_intercept=False, positive=False)
                model.fit(alg_positions, gt_positions)
                scale_factor = model.coef_
                offset = model.intercept_
                scaled_positions = np.dot(translation[:2], scale_factor.T) + offset
                x = scaled_positions[0]
                y = scaled_positions[1]

        elif detected.calculate_direction_change():
            scaled_positions = np.dot(translation[:2], scale_factor.T) + offset
            x = scaled_positions[0]
            y = scaled_positions[1]
        else:
            x = x / detected.get_scale_factor()
            y = y / detected.get_scale_factor()
            match detected.compare_total_directions():
                case 0:
                    ters_dizi = list(map(lambda pair: (pair[1], pair[0]), positions_data))
                    detected2 = Calculate_Direction(gt_data=gt_data, alg_data=ters_dizi)
                    match detected2.compare_total_directions():
                        case 1:
                            x = (x * -1) / detected.get_scale_factor()
                            y = (y) / detected.get_scale_factor()
                        case 2:
                            x = (x) / detected.get_scale_factor()
                            y = (y * -1)
                        case 3:
                            x = (x * -1) / detected.get_scale_factor()
                            y = (y * -1) / detected.get_scale_factor()

                case 1:
                    x = (x * -1) / detected.get_scale_factor()
                    y = (y) / detected.get_scale_factor()
                case 2:
                    x = (x) / detected.get_scale_factor()
                    y = (y * -1) / detected.get_scale_factor()
                case 3:
                    x = (x * -1) / detected.get_scale_factor()
                    y = (y * -1) / detected.get_scale_factor()

    print([x,y])

    '''detected_translation = [{
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
    if not os.path.exists("json"):
        os.makedirs("json")
    # JSON dosyasına yazma işlemi
    json_file_path = f"json/{name.split('.jpg')[0]}.json"  # Dilediğiniz dosya adını ve yolunu belirleyebilirsiniz
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
        print(f"JSON dosyası oluşturuldu: {json_file_path}")'''
    return x,y