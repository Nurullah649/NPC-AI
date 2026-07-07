import math


class DetectedTranslation:
    def __init__(self,
                 translation_x: float,
                 translation_y: float,
                 translation_z: float = 0.0):
        self.translation_x = self._sanitize(translation_x)
        self.translation_y = self._sanitize(translation_y)
        self.translation_z = self._sanitize(translation_z)

    @staticmethod
    def _sanitize(value, fallback=0.0):
        if value is None:
            return fallback
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return fallback
            return v
        except (TypeError, ValueError):
            return fallback

    def create_payload(self):
        return {
            'translation_x': str(self.translation_x),
            'translation_y': str(self.translation_y),
            'translation_z': str(self.translation_z),
        }
