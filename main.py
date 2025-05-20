import cv2
from tqdm import tqdm
from video_analyzer import VideoAnalyzer
from anomaly_detector import AnomalyDetector
from collections import Counter, defaultdict

# ---- Configurações iniciais ----
VIDEO_PATH = 'video.mp4'
anomaly_detector = AnomalyDetector(max_faces=2, max_missing_frames=5)

def process_video(video_path, exibir_frame=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Resolução: {frame_width}x{frame_height}, FPS: {fps}, Total de frames: {total_frames}")

    analyzer = VideoAnalyzer()
    anomaly_detector = AnomalyDetector()

    anomalies_detected = 0
    emotions_counter = Counter()
    emotions_frames = Counter()
    activities_counter = Counter()
    anomaly_frames = []
    anomaly_types_counter = Counter()
    anomaly_details = defaultdict(list)

    frames_info = []

    for frame_id in tqdm(range(total_frames), desc="Analisando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Análise do frame
        frame_marked, emotions, activity = analyzer.analyze_frame(frame)

        faces_count = len(emotions)
        frames_info.append({
            'frame_id': frame_id,
            'faces': faces_count,
            'emotion': emotions,
            'activity': activity,
        })

        # Atualização de contadores
        emotions_counter.update(emotions)
        activities_counter.update([activity])
        for e in set(emotions):
            emotions_frames[e] += 1

        # Preparar dados de faces para o detector de anomalias
        faces_data = [
            {'id': i, 'bbox': None, 'emotion': emotion}
            for i, emotion in enumerate(emotions)
        ]

        # === Detecção de anomalias ===
        anomalies = anomaly_detector.detect(frame_id, faces_data)
        if anomalies:
            anomaly_frames.append(frame_id)
            anomalies_detected += 1
            for a in anomalies:
                anomaly_types_counter[a] += 1
                anomaly_details[a].append(frame_id)

        # Exibir frame (opcional)
        if exibir_frame:
            cv2.imshow("Análise", frame_marked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Encerrar
    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

    metodologia = (
        "Metodologia aplicada:\n"
        "1. Reconhecimento Facial: Identificação e marcação dos rostos presentes em cada frame (Quando presentes e identificáveis baseado em um limiar).\n"
        "2. Análise de Expressões Emocionais: Para cada rosto detectado, análise da emoção predominante.\n"
        "3. Detecção de Atividades: Classificação da atividade predominante observada no frame.\n\n"

        "Critérios de anomalia utilizados:\n"
        "- Mais de 2 rostos no mesmo frame\n"
        "- Ausência de rostos por vários frames seguidos\n"
        "- Mudança brusca de emoção em uma mesma face"
    )

    report = {
        "Frames analizados": total_frames,
        "Anomalias detectadas": anomalies_detected,
        "Tipos de anomalias": dict(anomaly_types_counter),
        "Emoções mais comuns": emotions_counter.most_common(5),
        "Frames por emoção": dict(emotions_frames),
        "Atividades mais comuns": activities_counter.most_common(5),
        "Metodologia de Anomalia": metodologia
    }

    print("\n==== RELATÓRIO ====")
    for k, v in report.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v}")

    return report

# ---- Execução ----
if __name__ == "__main__":
    VIDEO_PATH = "video.mp4"  # Substitua com o caminho real
    process_video(VIDEO_PATH)
