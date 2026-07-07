class DetectedObject:
    def __init__(self, cls: int,
                 landing_status: str,
                 moving_status: str,
                 top_left_x: float,
                 top_left_y: float,
                 bottom_right_x: float,
                 bottom_right_y: float):
        self.cls = cls
        self.landing_status = landing_status
        self.moving_status = moving_status
        self.top_left_x = float(top_left_x)
        self.top_left_y = float(top_left_y)
        self.bottom_right_x = float(bottom_right_x)
        self.bottom_right_y = float(bottom_right_y)

    def create_payload(self, evaluation_server):
        # Normalize cls if it's a tuple
        cls_val = self.cls
        if isinstance(cls_val, (tuple, list)):
            cls_val = cls_val[0]
        try:
            cls_int = int(cls_val)
        except (TypeError, ValueError):
            cls_int = 0
        return {
            'cls': self._generate_api_url("classes/", str(cls_int + 1), evaluation_server),
            'landing_status': str(self.landing_status),
            'moving_status': str(self.moving_status),
            'top_left_x': str(int(self.top_left_x)),
            'top_left_y': str(int(self.top_left_y)),
            'bottom_right_x': str(int(self.bottom_right_x)),
            'bottom_right_y': str(int(self.bottom_right_y)),
        }

    @staticmethod
    def _generate_api_url(cls_endpoint, cls_id, evaluation_server):
        base = evaluation_server.rstrip("/")
        return f"{base}/{cls_endpoint}{cls_id}/"
