import cv2
import mediapipe as mp
import pickle
import numpy as np

print("Carregando modelo...")
with open('modelo_libras.pkl', 'rb') as f:
    modelo = pickle.load(f)

print("Modelo carregado!")

mp_h = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_estilo = mp.solutions.drawing_styles
detector = mp_h.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

camera = cv2.VideoCapture(0)
cv2.namedWindow("Reconhecimento Libras", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Reconhecimento Libras", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

texto_completo = ""
letra_atual = ""

while camera.isOpened():
    ok, frame = camera.read()
    if not ok:
        print("Erro na captura.")
        continue

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = detector.process(rgb)

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, mao, mp_h.HAND_CONNECTIONS,
                                   mp_estilo.get_default_hand_landmarks_style(),
                                   mp_estilo.get_default_hand_connections_style())

            pontos = [coord for ponto in mao.landmark for coord in (ponto.x, ponto.y, ponto.z)]
            predicao = modelo.predict([pontos])
            letra_atual = predicao[0]

            xs = [p.x for p in mao.landmark]
            ys = [p.y for p in mao.landmark]
            x_ini, y_ini = int(min(xs) * largura) - 10, int(min(ys) * altura) - 10
            x_fim, y_fim = int(max(xs) * largura) + 10, int(max(ys) * altura) + 10

            cv2.rectangle(frame, (x_ini, y_ini), (x_fim, y_fim), (0, 0, 0), 2)
            cv2.putText(frame, letra_atual, (x_ini, y_ini - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)

    cv2.putText(frame, f"Texto: {texto_completo}", (40, altura - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.imshow("Reconhecimento Libras", frame)

    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:  # ESC
        break
    elif tecla == 13:  # ENTER
        if letra_atual:
            texto_completo += letra_atual
            print("Adicionado:", texto_completo)
    elif tecla == 8:  # BACKSPACE
        texto_completo = texto_completo[:-1]

camera.release()
cv2.destroyAllWindows()
print("Encerrado.")
