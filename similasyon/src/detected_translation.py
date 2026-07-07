import math


class DetectedTranslation:
    def __init__(self,
                 translation_x: float,
                 translation_y: float,
                 translation_z: float,
                 ):

        self.translation_x = self._sanitize(translation_x)
        self.translation_y = self._sanitize(translation_y)
        self.translation_z = self._sanitize(translation_z)

    @staticmethod
    def _sanitize(value):
        """Sanitize NaN/Inf values to 0.0"""
        if value is None:
            return 0.0
        try:
            fval = float(value)
            if math.isnan(fval) or math.isinf(fval):
                return 0.0
            return fval
        except (TypeError, ValueError):
            return 0.0

    def create_payload(self):
        return {
                'translation_x': str(self.translation_x),
                'translation_y': str(self.translation_y),
                'translation_z': str(self.translation_z)
                }

