"""
Modulo 4 - Dispositivos de entrada e interacao
Demonstracao da ferramenta OpenCV

A camera vira um dispositivo de entrada: o programa le os quadros de video,
encontra um objeto colorido e devolve a posicao dele em pixels, do mesmo jeito
que um mouse devolve a posicao do cursor.

Como usar:
    python rastreador.py                 abre a webcam
    python rastreador.py exemplo.mp4     usa um video gravado, se a webcam falhar

Teclas:
    C   guarda a cor que estiver dentro do quadrado do meio
    Q   sai
"""

import sys
import cv2
import numpy as np

# 1. Abre a fonte de video. Sem argumento, usa a webcam (dispositivo 0).
fonte = sys.argv[1] if len(sys.argv) > 1 else 0
camera = cv2.VideoCapture(fonte)

if not camera.isOpened():
    print("Nao consegui abrir a camera. Feche outros programas que usam a webcam.")
    sys.exit(1)

cor_alvo = None      # matiz do objeto que estamos seguindo, em HSV
rastro = []          # ultimas posicoes, para desenhar o caminho percorrido

while True:
    ok, quadro = camera.read()
    if not ok:
        break

    quadro = cv2.flip(quadro, 1)          # espelha, para o movimento bater com o da pessoa
    altura, largura = quadro.shape[:2]

    # 2. O quadro de video e apenas uma matriz de numeros. Aqui ele e convertido
    # de BGR para HSV, onde a cor (matiz) fica separada do brilho, o que deixa o
    # reconhecimento muito menos sensivel a variacao de iluminacao.
    hsv = cv2.cvtColor(quadro, cv2.COLOR_BGR2HSV)

    if cor_alvo is None:
        # Ainda sem calibrar: mostra o alvo no centro e espera a tecla C
        cx, cy = largura // 2, altura // 2
        cv2.rectangle(quadro, (cx - 25, cy - 25), (cx + 25, cy + 25), (0, 232, 143), 2)
        cv2.putText(quadro, "Ponha o objeto no quadrado e aperte C",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # 3. Monta uma mascara: pixel branco onde a cor bate com a do objeto,
        # preto no resto. E uma imagem binaria, so zero e 255.
        margem = np.array([15, 80, 80])
        baixo = np.clip(cor_alvo - margem, 0, 255).astype(np.uint8)
        alto = np.clip(cor_alvo + margem, 0, 255).astype(np.uint8)
        mascara = cv2.inRange(hsv, baixo, alto)

        # Limpa pontinhos soltos causados por ruido da camera
        nucleo = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)

        # 4. Encontra as manchas brancas e fica com a maior delas
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            maior = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(maior) > 300:
                # O momento da mancha da o centro dela, que e a coordenada de saida
                m = cv2.moments(maior)
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])

                rastro.append((x, y))
                if len(rastro) > 40:
                    rastro.pop(0)

                cv2.circle(quadro, (x, y), 14, (0, 232, 143), 3)
                cv2.putText(quadro, "x={}  y={}".format(x, y),
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 232, 143), 2)

        # Desenha o caminho percorrido pelo objeto
        for i in range(1, len(rastro)):
            cv2.line(quadro, rastro[i - 1], rastro[i], (38, 159, 249), 2)

        cv2.imshow("Mascara (o que o computador enxerga)", mascara)

    cv2.imshow("Camera como dispositivo de entrada", quadro)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("c"):
        # Guarda a cor media do quadradinho central
        cx, cy = largura // 2, altura // 2
        recorte = hsv[cy - 25:cy + 25, cx - 25:cx + 25]
        cor_alvo = np.array(cv2.mean(recorte)[:3])
        rastro = []
        print("cor guardada (HSV):", cor_alvo.astype(int))
    elif tecla == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
