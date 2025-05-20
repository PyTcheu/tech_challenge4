import cv2
import mediapipe as mp
from emotion_detector import EmotionsDetector
from activity_detector import ActivityDetector

class VideoAnalyzer:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # MediaPipe modules
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        # Inicializa face detection, face mesh e pose
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_detection_confidence)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        # Inicializa detectores auxiliares
        self.emotion_detector = EmotionsDetector()
        self.activity_detector = ActivityDetector()

    def analyze_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- RECONHECIMENTO FACIAL ---
        face_results = self.face_detection.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                self.mp_drawing.draw_detection(frame, detection)

        # --- EXPRESSÕES FACIAIS ---
        emotions = []
        face_mesh_results = self.face_mesh.process(frame_rgb)
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                emotion = self.emotion_detector.detect(face_landmarks, frame.shape)
                emotions.append(emotion)
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        # --- DETECÇÃO DE ATIVIDADES ---
        activity = "Indefinida"
        pose_results = self.pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            activity = self.activity_detector.detect(pose_results.pose_landmarks, frame.shape)

        return frame, emotions, activity

    def close(self):
        self.face_detection.close()
        self.face_mesh.close()
        self.pose.close()
