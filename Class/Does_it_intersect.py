import os
from Class.ImageSimilaritryChecker import ImageSimilarityChecker

# UAP ve UAI inilebilir kontrolü yapacak olan modelin oluşturulması
image_similarity_checker = ImageSimilarityChecker()


def is_center_inside(human_box, obj_box):
    """
    Bir bounding box'un merkezinin başka bir bounding box'un içinde olup olmadığını kontrol eder.
    :param human_box: Merkezinin kontrol edileceği bounding box [x1, y1, x2, y2, score, class_id]
    :param obj_box: İçinde merkez olup olmadığı kontrol edilecek bounding box [x1, y1, x2, y2, score, class_id]
    :return: Merkez obj_box içindeyse True, değilse False
    """
    center_x = (human_box[0] + human_box[2]) / 2
    center_y = (human_box[1] + human_box[3]) / 2

    if obj_box[0] <= center_x <= obj_box[2] and obj_box[1] <= center_y <= obj_box[3]:
        return False
    else:
        return True


def does_human_center_intersect(results, path):
    """
    YOLO sonuçlarından cls 1 (İnsan) bounding box'unun merkezinin cls 2/3 bounding box'larının içinde olup olmadığını kontrol eder.

    :param results: YOLO sonuçları
    :return: Kesişme varsa True, yoksa False
    """
    for result in results:
        objects = result.boxes.data.tolist()

        # cls 1 (İnsan) ve cls 2/3 (UAP, UAI) kutuları filtreleme
        cls_1_boxes = [obj for obj in objects if obj[5] == 1]
        cls_2_3_boxes = [obj for obj in objects if obj[5] in [2, 3]]

        # Her bir insanın merkezi diğer objelerle kontrol edilir
        if cls_1_boxes:
            for human_box in cls_1_boxes:
                if any(is_center_inside(human_box, obj_box) for obj_box in cls_2_3_boxes):
                    for obj in objects:
                        x1, y1, x2, y2, _, class_id = obj
                        if class_id == 2 or class_id == 3:
                            return image_similarity_checker.control(x1=x1, y1=y1, x2=x2, y2=y2,
                                                                    image_path=os.path.join(path),
                                                                    class_id=class_id)
                else:
                    return False
        else:
            for obj in objects:
                x1, y1, x2, y2, _, class_id = obj
                if class_id == 2 or class_id == 3:
                    print(x1, y1, x2, y2)
                    return image_similarity_checker.control(x1=x1, y1=y1, x2=x2, y2=y2, image_path=os.path.join(path),
                                                            class_id=class_id)
