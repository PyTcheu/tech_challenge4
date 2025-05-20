import numpy as np
import mediapipe as mp

class ActivityDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def detect(self, pose_landmarks, frame_shape, landmark_history=None, prev_hand_positions=None):
        h, w, _ = frame_shape
        landmarks = pose_landmarks.landmark

        if self._is_lying_down(landmarks, h):
            print("Atividade no frame: Deitado")
            return "Deitado"
        elif self._is_sitting(landmarks, h):
            print("Atividade no frame: Sentado")
            return "Sentado"
        elif self._is_standing(landmarks, h):
            print("Atividade no frame: Em pé")
            return "Em pé"
        elif self._is_reading(landmarks, h, w):
            print("Atividade no frame: Lendo")
            return "Lendo"
        elif self._is_handling_object(landmarks, h, w, prev_hand_positions):
            print("Atividade no frame: Manuseando objeto")
            return "Manuseando objeto"
        elif landmark_history and self._is_dancing(landmark_history, w, h):
            print("Atividade no frame: Dançando")
            return "Dançando"
        else:
            print("Atividade no frame: Neutro")
            return "Neutro"

    def _is_lying_down(self, landmarks, h):
        y_coords = [
            landmarks[self.mp_pose.PoseLandmark.NOSE].y * h,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
        ]
        return max(y_coords) - min(y_coords) < 40  # tolerância pequena → quase no mesmo plano horizontal


    def _is_sitting(self, landmarks, h):
        shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h
        knee_y = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y * h
        vertical_ratio = abs(knee_y - hip_y) / abs(shoulder_y - hip_y + 1e-5)
        return vertical_ratio < 0.6  # pernas dobradas

    def _is_standing(self, landmarks, h):
        shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h
        knee_y = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y * h
        return abs(shoulder_y - hip_y) > 50 and abs(hip_y - knee_y) > 50

    def _is_reading(self, landmarks, h, w):
        # Cabeça inclinada + mãos na frente do tronco = possível leitura
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        hands_y_avg = (left_wrist.y + right_wrist.y) / 2 * h
        nose_y = nose.y * h
        shoulder_y = shoulder.y * h

        head_looking_down = nose_y > shoulder_y
        hands_near_torso = abs(left_wrist.y - shoulder.y) < 0.3 and abs(right_wrist.y - shoulder.y) < 0.3

        return head_looking_down and hands_near_torso

    def _is_handling_object(self, landmarks, h, w, prev_positions=None):
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        rw_x, rw_y = rw.x * w, rw.y * h
        rs_x, rs_y = rs.x * w, rs.y * h
        rh_y = rh.y * h

        in_zone = rs_y < rw_y < rh_y and abs(rw_x - rs_x) < 0.15 * w

        if prev_positions and len(prev_positions) >= 2:
            movement = np.linalg.norm(np.array(prev_positions[-1]) - np.array(prev_positions[-2]))
            return in_zone and movement < 15
        return in_zone

    def _is_dancing(self, landmark_history, w, h):
        movement_threshold = 0.05 * max(w, h)
        keypoints = [self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
                     self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        moving_count = 0
        for kp in keypoints:
            diffs = [
                np.linalg.norm(
                    np.array([landmark_history[i].landmark[kp.value].x * w,
                              landmark_history[i].landmark[kp.value].y * h]) -
                    np.array([landmark_history[i - 1].landmark[kp.value].x * w,
                              landmark_history[i - 1].landmark[kp.value].y * h])
                )
                for i in range(1, len(landmark_history))
            ]
            if np.mean(diffs) > movement_threshold:
                moving_count += 1
        return moving_count >= 2
