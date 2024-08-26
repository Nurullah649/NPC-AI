import rpy2.robjects as ro

def train_model():
    # r kodunu tetikler ve eğitim yapar
    r_script_path = "/home/nurullah/new_positioning_model/Positioning.R"
    # R script dosyasını çalıştırın
    ro.r['source'](r_script_path)

train_model()