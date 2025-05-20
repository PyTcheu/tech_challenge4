import numpy as np

class EmotionsDetector:
    def detect(self, landmarks, frame_shape):
        h, w, _ = frame_shape

        
        if self._is_sad(landmarks, w, h):
            print("Emoção no frame: Triste")
            return "Triste"
        elif self._is_angry(landmarks, w, h):
            print("Emoção no frame: Bravo")
            return "Bravo"
        elif self._is_anxious(landmarks, w, h):
            print("Emoção no frame: Ansioso")
            return "Ansioso"
        elif self._is_surprised(landmarks, w, h):
            print("Emoção no frame: Surpreso")
            return "Surpreso"
        elif self._is_distressed(landmarks, w, h):
            print("Emoção no frame: Angustiado")
            return "Angustiado"
        if self._is_happy(landmarks, w, h):
            print("Emoção no frame: Feliz")
            return "Feliz"
        else:
            print("Emoção no frame: Neutro")
            return "Neutro"
        
    def _is_happy(self, face_landmarks, w, h):
        landmarks = face_landmarks.landmark
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]

        mouth_width = np.linalg.norm([
            left_mouth.x * w - right_mouth.x * w,
            left_mouth.y * h - right_mouth.y * h
        ])
        mouth_height = np.linalg.norm([
            top_lip.x * w - bottom_lip.x * w,
            top_lip.y * h - bottom_lip.y * h
        ])
        ratio = mouth_height / mouth_width

        return ratio > 0.15
    
    def _is_sad(self, face_landmarks, w, h):
        lm = face_landmarks.landmark  # pega a lista indexável
        left_mouth = lm[61]
        right_mouth = lm[291]
        mid_mouth = lm[0]  # centro dos lábios
        avg_corner_y = (left_mouth.y + right_mouth.y) / 2
        center_y = mid_mouth.y
        return avg_corner_y > center_y + 0.25

    def _is_angry(self, face_landmarks, w, h):
        lm = face_landmarks.landmark
        left_brow = lm[70]
        right_brow = lm[300]
        left_eye = lm[159]
        right_eye = lm[386]
        brow_eye_dist_left = abs(left_brow.y - left_eye.y) * h
        brow_eye_dist_right = abs(right_brow.y - right_eye.y) * h
        return brow_eye_dist_left < 10 and brow_eye_dist_right < 10

    def _is_anxious(self, face_landmarks, w, h):
        lm = face_landmarks.landmark
        left_eye_top = lm[159]
        left_eye_bottom = lm[145]
        eye_height = abs(left_eye_top.y - left_eye_bottom.y) * h
        top_lip = lm[13]
        bottom_lip = lm[14]
        mouth_height = abs(top_lip.y - bottom_lip.y) * h
        return eye_height > 6 and mouth_height < 7
    
    def _is_surprised(self, face_landmarks, w, h):
        lm = face_landmarks.landmark
        
        # Distância vertical olho esquerdo (landmarks 159, 145)
        left_eye_height = abs(lm[159].y - lm[145].y) * h
        # Distância sobrancelha esquerda ao olho (landmarks 70 e 159)
        left_brow_eye_dist = abs(lm[70].y - lm[159].y) * h
        # Distância boca aberta (lábios 13 e 14)
        mouth_open = abs(lm[13].y - lm[14].y) * h
        # Critérios: olho aberto, sobrancelha alta, boca aberta
        return (left_eye_height > 12 and 
                left_brow_eye_dist > 18 and 
                mouth_open > 5)

    def _is_distressed(self, face_landmarks, w, h):
        lm = face_landmarks.landmark
        
        # Distância entre sobrancelhas (landmarks 70 e 300)
        brow_distance = abs(lm[70].x - lm[300].x) * w
        
        # Distância olho aberto (usar olho esquerdo como referência)
        eye_height = abs(lm[159].y - lm[145].y) * h
        
        # Altura boca (lábios)
        mouth_height = abs(lm[13].y - lm[14].y) * h
        
        # Critérios: sobrancelhas próximas, olhos pouco abertos, boca fechada (lábios juntos)
        return (brow_distance < 0.05 and 
                eye_height < 6 and 
                mouth_height < 4)