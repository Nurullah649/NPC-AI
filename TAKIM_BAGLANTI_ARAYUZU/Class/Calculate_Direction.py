import numpy as np
import pandas as pd

class Calculate_Direction:
    def __init__(self, gt_data=None, alg_data=None):
        self.gt_data = gt_data
        self.alg_data = alg_data
        self.direction_changes_value = self.calculate_direction_change()
        self.gt_to_alg_direction = self.compare_total_directions()
        self.scale_factor = self.calculate_scale_factor()

    def calculate_direction_change(self):
        direction_changes = 0
        # Başlangıç yönünü belirle
        previous_vector = np.array(self.gt_data[1]) - np.array(self.gt_data[0])

        for i in range(1, len(self.gt_data) - 1):
            current_vector = np.array(self.gt_data[i + 1]) - np.array(self.gt_data[i])

            # İki vektör arasındaki açıya bakıyoruz (dot product kullanarak)
            if np.dot(previous_vector, current_vector) < 0:
                direction_changes += 1

            previous_vector = current_vector
        if direction_changes >= 1:
            return True
        else:
            return False

    def compare_total_directions(self):
        # Tüm noktalar için toplam vektörleri hesapla
        total_gt_vector = np.sum(np.diff(self.gt_data, axis=0), axis=0)
        total_alg_vector = np.sum(np.diff(self.alg_data, axis=0), axis=0)

        # Normalize ederek yön vektörünü birim vektöre çeviriyoruz
        gt_direction = total_gt_vector / np.linalg.norm(total_gt_vector)
        alg_direction = total_alg_vector / np.linalg.norm(total_alg_vector)

        # Dot product ile yönlerin ne kadar örtüştüğünü buluyoruz
        direction_similarity = np.dot(gt_direction, alg_direction)

        # X ve Y eksenlerinde ters yön kontrolü
        x_similarity = np.sign(gt_direction[0]) == np.sign(alg_direction[0])
        y_similarity = np.sign(gt_direction[1]) == np.sign(alg_direction[1])

        if direction_similarity < 0:
            return 0
        else:
            if not x_similarity and y_similarity:
                return 1
            elif x_similarity and not y_similarity:
                return 2
            elif not x_similarity and not y_similarity:
                return 3
            else:
                return 4

    def calculate_scale_factor(self):
        # Her iki veri seti arasındaki ölçek farkını hesaplar
        total_gt_vector = np.sum(np.diff(self.gt_data, axis=0), axis=0)
        total_alg_vector = np.sum(np.diff(self.alg_data, axis=0), axis=0)

        # İki vektörün normunu hesapla
        gt_norm = np.linalg.norm(total_gt_vector)
        alg_norm = np.linalg.norm(total_alg_vector)

        # Ölçek faktörünü hesapla
        scale_factor = alg_norm / gt_norm

        return scale_factor

    def get_direction_changes_value(self):
        return self.direction_changes_value

    def get_gt_to_alg_direction(self):
        return self.gt_to_alg_direction

    def get_scale_factor(self):
        return self.scale_factor
