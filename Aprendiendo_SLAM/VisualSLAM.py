import cv2
import numpy as np
import open3d as o3d

# === Configuración de cámara ficticia ===
K = np.array([[700, 0, 320],
              [0, 700, 240],
              [0,   0,   1]])

# === Inicialización ===
cap = cv2.VideoCapture("video.mp4")
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Variables de SLAM
poses = [np.eye(4)]
puntos_3d = []

# Visualizador Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Reconstrucción 3D")
cloud = o3d.geometry.PointCloud()
vis.add_geometry(cloud)

def actualizar_nube(puntos_3d):
    cloud.points = o3d.utility.Vector3dVector(np.array(puntos_3d))
    vis.update_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()

# === Bucle principal ===
ret, frame_anterior = cap.read()
gris_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(gris_anterior, None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gris, None)
    if des1 is None or des2 is None:
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

    nueva_pose = np.eye(4)
    nueva_pose[:3, :3] = R
    nueva_pose[:3, 3] = t[:, 0]
    poses.append(poses[-1] @ np.linalg.inv(nueva_pose))

    for i, m in enumerate(matches):
        if mask_pose[i] == 0:
            continue
        z = 1.0
        x = (pts2[i][0] - K[0, 2]) * z / K[0, 0]
        y = (pts2[i][1] - K[1, 2]) * z / K[1, 1]
        punto = np.array([x, y, z, 1.0])
        mundo = poses[-1] @ punto
        puntos_3d.append(mundo[:3])

    actualizar_nube(puntos_3d)

    img_matches = cv2.drawMatches(gris_anterior, kp1, gris, kp2, matches, None, flags=2)
    cv2.imshow("Video con puntos", img_matches)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    kp1, des1 = kp2, des2
    gris_anterior = gris

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

# Guardar nube
nube_final = o3d.geometry.PointCloud()
nube_final.points = o3d.utility.Vector3dVector(np.array(puntos_3d))
o3d.io.write_point_cloud("point_cloud.ply", nube_final)
print(" Nube de puntos guardada en point_cloud.ply")
