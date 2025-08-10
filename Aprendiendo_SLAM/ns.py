import cv2
import numpy as np
import open3d as o3d
import time

# -------------------------
# Configuración cámara (intrínsecos)
# -------------------------
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0,   0,   1]], dtype=np.float64)

# Parámetros
VIDEO_PATH = "video.mp4"
MAX_MATCHES = 300
MIN_MATCHES_FOR_RECOVER = 30
RATIO_TEST = 0.75
REPROJ_THRESH = 3.0  # px reprojection error threshold

# -------------------------
# Inicialización
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("No se pudo abrir el video.")

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # usaremos knn + ratio test

# Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Nube 3D (triangulación)", width=900, height=700)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Variables de estado
puntos_3d_acumulados = []
poses = [np.eye(4, dtype=np.float64)]  # poses acumuladas (pose del frame0)

# Leer primer frame
ret, frame_prev = cap.read()
if not ret:
    raise IOError("No se pudo leer el primer frame.")
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
kp_prev = orb.detect(gray_prev, None)
kp_prev, des_prev = orb.compute(gray_prev, kp_prev)

frame_idx = 1
t0 = time.time()

def to_homogeneous(pts):
    """Convierte Nx3 a 4xN homogénea (para multiplicación)."""
    return np.vstack((pts.T, np.ones((1, pts.shape[0]))))

def triangulate_and_filter(P0, P1, pts0, pts1, K, pose_prev, pose_curr):
    """
    Triangula correspondencias (Nx2 arrays pts0, pts1).
    Devuelve lista de puntos 3D en coordenadas del mundo (pose_prev ya aplicado).
    Filtra por Z>0 y reproyección.
    """
    if len(pts0) < 2:
        return []

    # Triangulación (devuelve homogéneas 4xN)
    pts4d_h = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)  # 4xN
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # Nx3

    pts_world = []
    # Comprobar reproyección y Z>0
    P0_full = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P1_full = K @ np.hstack((pose_curr[:3,:3], pose_curr[:3,3:4]))  # pose_curr is [R|t] in world? careful
    # We'll test reprojection into current frame (pts1)
    for i, p in enumerate(pts3d):
        # Transform point to camera coords of current frame
        # Since pts3d are in camera0 coords; we want camera1 coords:
        # X_cam1 = R * X_cam0 + t  where R,t = transformation from cam0 to cam1 (pose_rel)
        # But cv2.recoverPose gave R,t from cam0->cam1 so we can reuse that.
        # Here we assume pts3d computed in cam0 frame. We'll reproj into cam1:
        Xc1 = P1[:3,:3] @ np.array([p[0], p[1], p[2]]) + P1[:3,3]
        if Xc1[2] <= 0:
            continue

        # Reproject into image 1
        x_proj = (K @ Xc1)[0:2] / (K @ Xc1)[2]
        err = np.linalg.norm(x_proj.flatten() - pts1[i].flatten())
        if err > REPROJ_THRESH:
            continue

        # Transform point from cam0 coordinates to world using pose_prev (world = pose_prev * cam0_point)
        cam0_point_h = np.array([p[0], p[1], p[2], 1.0])
        world_point = pose_prev @ cam0_point_h  # assuming pose_prev maps cam0 -> world
        pts_world.append(world_point[:3])

    return pts_world

# Nota: necesitamos definir bien poses: poses[i] = transform that maps points in camera_i coords to world coords.
# Inicialmente poses[0] = I (cam0 == world). Cuando obtenemos R,t de recoverPose entre cam0->cam1,
# la transformacion de cam1 en world: pose1 = pose0 @ inv([R|t])  OR depending sign conventions.
# En este código usaremos la aproximación: pose_curr = pose_prev @ np.linalg.inv(T_rel),
# donde T_rel = [[R, t],[0,1]] transforma puntos de cam0 a cam1.


while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr = orb.detect(gray, None)
    kp_curr, des_curr = orb.compute(gray, kp_curr)

    if des_prev is None or des_curr is None:
        # actualizar y continuar
        gray_prev = gray.copy()
        kp_prev, des_prev = kp_curr, des_curr
        frame_prev = frame.copy()
        frame_idx += 1
        continue

    # Emparejar con KNN + ratio test (más robusto que crossCheck True)
    matches_knn = bf.knnMatch(des_prev, des_curr, k=2)
    good_matches = []
    pts_prev = []
    pts_curr = []
    for m_n in matches_knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)
            pts_prev.append(kp_prev[m.queryIdx].pt)
            pts_curr.append(kp_curr[m.trainIdx].pt)

    if len(good_matches) < MIN_MATCHES_FOR_RECOVER:
        # No suficientes matches, actualizar y continuar
        gray_prev = gray.copy()
        kp_prev, des_prev = kp_curr, des_curr
        frame_prev = frame.copy()
        frame_idx += 1
        continue

    pts_prev = np.array(pts_prev, dtype=np.float64)
    pts_curr = np.array(pts_curr, dtype=np.float64)

    # Estimar essential matrix y pose relativa
    E, maskE = cv2.findEssentialMat(pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        gray_prev = gray.copy()
        kp_prev, des_prev = kp_curr, des_curr
        frame_prev = frame.copy()
        frame_idx += 1
        continue

    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=maskE)

    # Construir T_rel (4x4) que transforma puntos de cam0 a cam1: X1 = R_rel * X0 + t_rel
    T_rel = np.eye(4, dtype=np.float64)
    T_rel[:3,:3] = R_rel
    T_rel[:3,3] = t_rel.ravel()

    # Definir poses: pose_prev maps cam0 -> world. Calculamos pose_curr maps cam1 -> world:
    # pose_curr * X_cam1 = world. Y como X_cam1 = T_rel * X_cam0 -> pose_curr = pose_prev * inv(T_rel)
    pose_prev = poses[-1]
    try:
        T_rel_inv = np.linalg.inv(T_rel)
    except np.linalg.LinAlgError:
        T_rel_inv = np.eye(4)
    pose_curr = pose_prev @ T_rel_inv
    poses.append(pose_curr)

    # Matrices de proyección: P0 = K [I|0], P1 = K [R_rel | t_rel] but in camera0 coords.
    P0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K @ np.hstack((R_rel, t_rel))

    # Sólo usar matches que son inliers por mask_pose (mask_pose shape Nx1)
    mask_pose = mask_pose.ravel()
    inlier_idx = np.where(mask_pose == 1)[0]
    if len(inlier_idx) > 0:
        pts0_in = pts_prev[inlier_idx].reshape(-1,2)
        pts1_in = pts_curr[inlier_idx].reshape(-1,2)

        # Triangula y filtra
        puntos_world = triangulate_and_filter(P0, P1, pts0_in, pts1_in, K, pose_prev, T_rel)
        # NOTA: triangulate_and_filter devuelve puntos transformados por pose_prev (world coords)

        for p in puntos_world:
            puntos_3d_acumulados.append(p)

    # Visualizar video con matches (dibujar líneas)
    drawn = cv2.drawMatches(frame_prev, kp_prev, frame, kp_curr, [good_matches[i] for i in inlier_idx], None, flags=2)
    cv2.imshow("Video - matches inliers", drawn)

    # Actualizar nube Open3D
    if len(puntos_3d_acumulados) > 0:
        pts_np = np.array(puntos_3d_acumulados, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.8,0.1,0.1]), (pts_np.shape[0],1)))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Preparar siguiente iteración
    gray_prev = gray.copy()
    kp_prev, des_prev = kp_curr, des_curr
    frame_prev = frame.copy()
    frame_idx += 1

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

# Guardar nube resultante (recuerda: mapa en escala arbitraria)
if len(puntos_3d_acumulados) > 0:
    nube_final = o3d.geometry.PointCloud()
    nube_final.points = o3d.utility.Vector3dVector(np.array(puntos_3d_acumulados))
    o3d.io.write_point_cloud("triangulated_map.ply", nube_final)
    print("Mapa 3D guardado en triangulated_map.ply (escala arbitraria)")
else:
    print("No se generaron puntos 3D sólidos.")
