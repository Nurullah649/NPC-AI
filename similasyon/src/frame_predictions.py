class FramePredictions:
    def __init__(self, frame_url, image_url, video_name,
                 gt_translation_x=None, gt_translation_y=None, gt_translation_z=None):
        self.frame_url = frame_url
        self.image_url = image_url
        self.video_name = video_name
        self.gt_translation_x = gt_translation_x
        self.gt_translation_y = gt_translation_y
        self.gt_translation_z = gt_translation_z
        self.detected_objects = []
        self.detected_translations = []
        self.reference_predictions = []

    def add_detected_object(self, detection):
        self.detected_objects.append(detection)

    def add_translation_object(self, translation):
        self.detected_translations.append(translation)

    def add_reference_prediction(self, ref_pred):
        self.reference_predictions.append(ref_pred)

    def create_payload(self, evaluation_server):
        payload = {
            "frame": self.frame_url,
            "detected_objects": self._create_objects_payload(evaluation_server),
            "detected_translations": self._create_translations_payload(),
            "reference_predictions": self._create_ref_predictions_payload(),
        }
        return payload

    def _create_objects_payload(self, evaluation_server):
        return [d_obj.create_payload(evaluation_server) for d_obj in self.detected_objects]

    def _create_translations_payload(self):
        return [t_obj.create_payload() for t_obj in self.detected_translations]

    def _create_ref_predictions_payload(self):
        return [r.create_payload() for r in self.reference_predictions]
