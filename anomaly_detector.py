class AnomalyDetector:
    def __init__(self, max_faces=2, max_missing_frames=5):
        self.max_faces = max_faces
        self.max_missing_frames = max_missing_frames
        
        self.frames_since_last_face = 0
        self.last_emotions = {}

    def reset(self):
        self.frames_since_last_face = 0
        self.last_emotions.clear()
    
    def detect(self, frame_id, faces_data):
        anomalies = []

        # 1. Muitas faces no frame
        if len(faces_data) > self.max_faces:
            msg = f"muitas_faces ({len(faces_data)})"
            print(f"[Frame {frame_id}] Anomalia: {msg}")
            anomalies.append("muitas_faces")

        # 2. Nenhum rosto detectado por muitos frames seguidos
        if len(faces_data) == 0:
            self.frames_since_last_face += 1
            if self.frames_since_last_face > self.max_missing_frames:
                msg = f"rostos_ausentes_{self.frames_since_last_face}_frames"
                print(f"[Frame {frame_id}] Anomalia: {msg}")
                anomalies.append("rostos_ausentes")
        else:
            self.frames_since_last_face = 0

        # 3. Mudança brusca de emoção
        for face in faces_data:
            face_id = face['id']
            emotion = face['emotion']

            if face_id in self.last_emotions:
                if self.last_emotions[face_id] != emotion:
                    print(f"[Frame {frame_id}] Anomalia: mudança_emocao_face_{face_id} ({self.last_emotions[face_id]} -> {emotion})")
                    anomalies.append("mudanca_rapida_emocao")
            self.last_emotions[face_id] = emotion

        return anomalies
