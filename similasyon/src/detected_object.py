class DetectedObject:
    def __init__(self, cls: int,
                 landing_status: str,
                 moving_status: str,
                 top_left_x: float,
                 top_left_y: float,
                 bottom_right_x: float,
                 bottom_right_y: float,
                 track_id=None,
                 _motion_score=None,
                 ):
        self.cls = cls
        self.landing_status = str(landing_status)
        self.moving_status = str(moving_status)
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.track_id = track_id
        self._motion_score = _motion_score

    def create_payload(self, evaluation_server):
        # Official: cls is a tuple (classes["Tasit"],) so we index [0]
        cls_val = self.cls
        if isinstance(cls_val, (tuple, list)):
            cls_val = cls_val[0]
        try:
            cls_int = int(cls_val)
        except (TypeError, ValueError):
            cls_int = 0
        return {
            'cls': self.generate_api_url("classes/", str(cls_int + 1), evaluation_server),
            'landing_status': str(self.landing_status),
            'moving_status': str(self.moving_status),
            'top_left_x': str(self.top_left_x),
            'top_left_y': str(self.top_left_y),
            'bottom_right_x': str(self.bottom_right_x),
            'bottom_right_y': str(self.bottom_right_y),
        }

    @staticmethod
    def generate_api_url(cls_endpoint, cls_id, evaluation_server):
        checked_url = evaluation_server if evaluation_server[-1] != "/" else evaluation_server + "/"
        return evaluation_server + cls_endpoint + cls_id + "/"
