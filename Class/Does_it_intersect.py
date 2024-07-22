def Calculate_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
def does_it_intersect(results):
    for result in results:
        objects = result.boxes.xyxy.tolist()
        cls_objects=result.boxes.cls.tolist()
        id_objects=result.boxes.id.tolist()
        length=len(objects)
        for i in range(length):
            for j in range(i+1,length):

                if cls_objects[i]==0 and cls_objects[j]==1:
                    print(id_objects[i], " ", id_objects[j], " ", Calculate_IOU(objects[i], objects[j]))
                    if Calculate_IOU(objects[i],objects[j])>0.5:
                        print("Intersection")
                        objects.delete(objects[j])
                        return objects
                elif cls_objects[i]==1 and cls_objects[j]==0:
                    print(id_objects[i], " ", id_objects[j], " ", Calculate_IOU(objects[i], objects[j]))
                    if Calculate_IOU(objects[i],objects[j])>0.5:
                        print("Intersection")
                        objects.delete(objects[j])
                        return objects
    return objects