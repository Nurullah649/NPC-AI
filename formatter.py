import json

def create_json_data(id, user, frame, detected_objects, detected_translations):
    data = {
        "id": id,
        "user": user,
        "frame": frame,
        "detected_objects": detected_objects,
        "detected_translations": detected_translations
    }
    return json.dumps(data, indent=2)
