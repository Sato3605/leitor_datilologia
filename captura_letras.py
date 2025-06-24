import cv2
import mediapipe as mp
import csv
import os

# --- Inicialização ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

arquivo_csv = 'dados_libras.csv'
total_amostras = 200

# Cabeçalho do CSV
colunas = ['letra']
for i in range(21):
    colunas += [f'x{i}', f'y{i}', f'z{i}']

# Cria CSV se ainda não existir
if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, 'w', newline='') as f:
        csv.writer(f).writerow(colunas)

# --- Captura da Webcam ---
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        print("Erro ao acessar câmera.")
        break

    frame = cv2.flip(frame, 1)
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands_detector.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 100, 255), thickness=2)
            )

    cv2.imshow('Coleta de Gestos - Libras', frame)
    tecla = cv2.waitKey(10) & 0xFF

    if tecla == ord('0'):
        break

    if ord('a') <= tecla <= ord('z'):
        letra = chr(tecla)
        print(f"\nColetando dados da letra: {letra.upper()}")

        for i in range(total_amostras):
            sucesso, coleta = camera.read()
            if not sucesso:
                continue

            coleta = cv2.flip(coleta, 1)
            coleta_rgb = cv2.cvtColor(coleta, cv2.COLOR_BGR2RGB)
            resultado_coleta = hands_detector.process(coleta_rgb)

            if resultado_coleta.multi_hand_landmarks:
                for mao in resultado_coleta.multi_hand_landmarks:
                    linha = [letra] + [coord for lm in mao.landmark for coord in (lm.x, lm.y, lm.z)]
                    with open(arquivo_csv, 'a', newline='') as f:
                        csv.writer(f).writerow(linha)

            status = f"Capturando: {i+1}/{total_amostras}"
            cv2.putText(coleta, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Coleta de Gestos - Libras', coleta)
            cv2.waitKey(1)

        print(f"Coleta de '{letra.upper()}' concluída.")

camera.release()
cv2.destroyAllWindows()
