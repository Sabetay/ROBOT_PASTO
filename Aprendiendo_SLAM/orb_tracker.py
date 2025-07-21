import cv2
# Importa la biblioteca OpenCV, que se usa para procesamiento de imágenes y visión por computadora.

# Cargar el video
cap = cv2.VideoCapture("video.mp4")
# Abre el archivo de video "video.mp4" para capturar sus fotogramas.

# Crear detector ORB
orb = cv2.ORB_create(nfeatures=1000)
# Crea un detector ORB (Oriented FAST and Rotated BRIEF) para detectar hasta 1000 características por imagen.

while cap.isOpened():
    # Bucle principal: se ejecuta mientras el video esté abierto y disponible.

    ret, frame = cap.read()
    # Lee el siguiente fotograma del video. 'ret' es True si la lectura fue exitosa, 'frame' contiene la imagen.

    if not ret:
        break
    # Si no se pudo leer un fotograma (fin del video o error), sale del bucle.

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convierte el fotograma de color (BGR) a escala de grises, necesario para el detector ORB.

    # Detectar puntos clave y descriptores
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    # Detecta los puntos clave (keypoints) y calcula sus descriptores en la imagen en escala de grises.

    # Dibujar los puntos clave
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    # Dibuja los puntos clave detectados sobre el fotograma original, usando color verde.

    cv2.imshow('ORB Feature Tracking', frame_with_keypoints)
    # Muestra el fotograma con los puntos clave en una ventana llamada 'ORB Feature Tracking'.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Espera 1 milisegundo por una tecla. Si se presiona 'q', sale del bucle.

cap.release()
# Libera el recurso de captura de video.

cv2.destroyAllWindows()
# Cierra todas las ventanas abiertas por OpenCV.