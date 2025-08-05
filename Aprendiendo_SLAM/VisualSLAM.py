import cv2
import numpy as np
import open3d as o3d

# Parámetros de la cámara (debes calibrar la tuya para valores precisos)
focal = 700  # píxeles, aproximado
pp = (320, 240)  # punto principal (cx, cy), ejemplo

def extraer_kp_desc(img, orb):
    """Extrae keypoints y descriptores usando ORB."""
    return orb.detectAndCompute(img, None)

def emparejar_descriptores(desc1, desc2):
    """Empareja descriptores usando BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

def obtener_puntos(kp1, kp2, matches):
    """Obtiene puntos clave emparejados."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

# Inicialización de captura de video y ORB
cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    print("No se pudo abrir el video. Verifica la ruta o el archivo.")
    exit()

orb = cv2.ORB_create(2000)

# Leer el primer cuadro
ret, frame_anterior = cap.read()
if not ret:
    print("No se pudo leer el primer cuadro del video.")
    exit()

gris_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
kp1, desc1 = extraer_kp_desc(gris_anterior, orb)

# Inicialización de variables para SLAM
trayectoria = [np.array([0, 0, 0])]
R_total = np.eye(3)
t_total = np.zeros((3, 1))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = extraer_kp_desc(gris, orb)

    # Validación de descriptores
    if desc2 is None or desc1 is None or len(desc2) < 10 or len(desc1) < 10:
        desc1, kp1 = desc2, kp2
        continue

    matches = emparejar_descriptores(desc1, desc2)
    if len(matches) < 8:
        desc1, kp1 = desc2, kp2
        continue

    pts1, pts2 = obtener_puntos(kp1, kp2, matches)
    E, mask = cv2.findEssentialMat(pts2, pts1, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        desc1, kp1 = desc2, kp2
        continue

    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, focal=focal, pp=pp)
    t_total += R_total.dot(t)
    R_total = R.dot(R_total)
    trayectoria.append(t_total.flatten())

    desc1, kp1 = desc2, kp2

    # Dibujo de coincidencias para visualizar en tiempo real
    num_matches = min(len(matches), 50)
    frame_matches = cv2.drawMatches(gris_anterior, kp1, gris, kp2, matches[:num_matches], None, flags=2)
    cv2.imshow("SLAM - Coincidencias ORB", frame_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupción manual.")
        break

cap.release()
trayectoria = np.array(trayectoria)

# Guardar la trayectoria estimada
np.save("trayectoria_slam.npy", trayectoria)

# Visualización 3D
pts = o3d.utility.Vector3dVector(trayectoria)
nube = o3d.geometry.PointCloud()
nube.points = pts
nube.paint_uniform_color([1.0, 0.0, 0.0])  # rojo

o3d.visualization.draw_geometries([nube], window_name="Trayectoria SLAM")

cv2.destroyAllWindows()