"""
SCRIPT 4 - CALIBRATION HAND-EYE
Relie automatiquement le repère caméra iPhone au repère robot SO-101.
Le robot bouge à plusieurs positions connues pendant que la caméra observe
un marqueur coloré sur la pince.

AVANT DE LANCER :
- La calibration caméra doit être faite (camera_params.npz doit exister)
- Le robot doit être connecté sur /dev/ttyACM1
- Placer un objet coloré (rouge ou autre) SUR la pince du robot
- L'iPhone doit voir toute la zone de travail
"""

import cv2
import numpy as np
import time
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig as SO101FollowerConfig

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
PARAMS_FILE     = "camera_params.npz"
OUTPUT_FILE     = "handeye_params.npz"
ROBOT_PORT      = "/dev/ttyACM4"
CAMERA_HEIGHT_CM = 30.0   # hauteur iPhone au-dessus de la table en cm

# Couleur du marqueur sur la pince (rouge par défaut)
HSV_LOW  = np.array([0, 100, 150])
HSV_HIGH = np.array([15, 255, 255])

# Positions connues du robot pour la calibration (angles en degrés)
# Le robot va aller à chacune de ces positions, on détecte la pince dans l'image
CALIBRATION_POSES = [
    # Positions sûres basées sur la position de repos réelle du robot
    # repos mesuré : shoulder_pan=-1.6, shoulder_lift=-99.7, elbow_flex=99.0, wrist_flex=27.9
    # On garde le bras replié (shoulder_lift bas, elbow_flex élevé) et on varie shoulder_pan
    {'shoulder_pan':   0, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':  25, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan': -25, 'shoulder_lift': -90, 'elbow_flex': 90, 'wrist_flex': 30, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':  15, 'shoulder_lift': -80, 'elbow_flex': 85, 'wrist_flex': 25, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan': -15, 'shoulder_lift': -80, 'elbow_flex': 85, 'wrist_flex': 25, 'wrist_roll': 0, 'gripper': 10},
    {'shoulder_pan':   0, 'shoulder_lift': -85, 'elbow_flex': 88, 'wrist_flex': 28, 'wrist_roll': 0, 'gripper': 10},
]
# ──────────────────────────────────────────────────────────────────────────────

# Cinématique directe SO-101 (depuis URDF officiel)
def Rx(deg):
    r = np.deg2rad(deg)
    return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])

def Ry(deg):
    r = np.deg2rad(deg)
    return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])

def Rz(deg):
    r = np.deg2rad(deg)
    return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])

def make_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def forward_kinematics(pose):
    """Retourne la position (x,y,z) de la pince en mètres dans le repère robot."""
    T_w1 = make_T(Rz(180) @ Rx(180) @ Rz(pose['shoulder_pan']),
                  [0.0388353, 0.0, 0.0624])
    T_12 = make_T(Rx(90) @ Ry(pose['shoulder_lift']),
                  [0.0, -0.0452, 0.0])
    T_23 = make_T(Ry(pose['elbow_flex']),
                  [0.1013, 0.0, 0.0])
    T_34 = make_T(Rx(-90) @ Ry(pose['wrist_flex']),
                  [0.094, 0.0, 0.0])
    T_45 = make_T(Rx(90) @ Rz(pose['wrist_roll']),
                  [0.0, -0.0426, 0.0])
    T_5t = make_T(np.eye(3), [0.0, 0.0, 0.065])

    T = T_w1 @ T_12 @ T_23 @ T_34 @ T_45 @ T_5t
    return T[:3, 3]  # position x, y, z en mètres

def detect_marker(frame, K, dist):
    """Détecte le marqueur coloré et retourne sa position (u,v) en pixels."""
    frame_ud = cv2.undistort(frame, K, dist)
    hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame_ud
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None, frame_ud
    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    cv2.circle(frame_ud, (u,v), 8, (0,255,0), -1)
    cv2.putText(frame_ud, f"Marqueur ({u},{v})", (u+10, v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return (u, v), frame_ud

# ── Chargement calibration caméra ──
try:
    data = np.load(PARAMS_FILE)
    K    = data["K"]
    dist = data["dist"]
    print("Calibration caméra chargée.")
except FileNotFoundError:
    print(f"'{PARAMS_FILE}' introuvable. Lance d'abord 2_calibration.py")
    exit()

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

# ── Connexion robot ──
print(f"Connexion au robot sur {ROBOT_PORT}...")
config = SO101FollowerConfig(port=ROBOT_PORT)
robot  = SO101Follower(config)
robot.connect()
print("Robot connecté.")

# ── Caméra ──
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

robot_positions_m = []   # positions pince dans repère robot (mètres)
camera_positions_cm = [] # positions pince dans repère caméra (cm)

print(f"\nDébut calibration hand-eye : {len(CALIBRATION_POSES)} positions")
print("Appuie sur ESPACE pour valider chaque position, Q pour annuler.\n")

for i, pose in enumerate(CALIBRATION_POSES):
    print(f"Position {i+1}/{len(CALIBRATION_POSES)} — déplacement du robot...")

    # Envoyer la consigne au robot
    action = {f"{k}.pos": v for k, v in pose.items()}
    robot.send_action(action)
    time.sleep(2.0)  # attendre que le robot soit stable

    # Position pince en cinématique directe
    pos_robot = forward_kinematics(pose)  # en mètres
    print(f"  FK robot : x={pos_robot[0]*100:.1f}cm  y={pos_robot[1]*100:.1f}cm  z={pos_robot[2]*100:.1f}cm")

    # Capture et détection du marqueur
    detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        uv, display = detect_marker(frame, K, dist)

        status = f"Pos {i+1}/{len(CALIBRATION_POSES)} — "
        status += "MARQUEUR DETECTE — ESPACE pour valider" if uv else "Marqueur non detecte"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if uv else (0,0,255), 2)
        cv2.imshow("Hand-Eye Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and uv:
            # Convertir pixels → cm dans repère caméra
            u, v_px = uv
            Z = CAMERA_HEIGHT_CM
            X_cam = (u  - cx) * Z / fx
            Y_cam = (v_px - cy) * Z / fy
            camera_positions_cm.append([X_cam, Y_cam, Z])
            robot_positions_m.append(pos_robot)
            print(f"  Cam      : X={X_cam:.1f}cm  Y={Y_cam:.1f}cm  Z={Z:.1f}cm  ✓")
            detected = True
            break
        elif key == ord('q'):
            print("Annulé.")
            cap.release()
            cv2.destroyAllWindows()
            robot.disconnect()
            exit()

cv2.destroyAllWindows()
cap.release()

# ── Calcul de la transformation caméra → robot ──
print("\nCalcul de la transformation hand-eye...")

cam_pts  = np.array(camera_positions_cm)   # N x 3, en cm
rob_pts  = np.array(robot_positions_m) * 100  # N x 3, en cm

# Résolution par moindres carrés : rob = R @ cam + t
# On utilise la méthode SVD pour trouver R et t
cam_mean = cam_pts.mean(axis=0)
rob_mean = rob_pts.mean(axis=0)

cam_c = cam_pts - cam_mean
rob_c = rob_pts - rob_mean

H = cam_c.T @ rob_c
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T

# Correction si det(R) = -1 (réflexion)
if np.linalg.det(R) < 0:
    Vt[-1,:] *= -1
    R = Vt.T @ U.T

t = rob_mean - R @ cam_mean

print(f"\nTransformation calculée :")
print(f"  Rotation R :\n{R}")
print(f"  Translation t : {t} cm")

# Vérification : erreur de reprojection
errors = []
for i in range(len(cam_pts)):
    pred = R @ cam_pts[i] + t
    err  = np.linalg.norm(pred - rob_pts[i])
    errors.append(err)
    print(f"  Point {i+1} : erreur = {err:.2f} cm")

print(f"\nErreur moyenne : {np.mean(errors):.2f} cm")

# Sauvegarde
np.savez(OUTPUT_FILE, R=R, t=t)
print(f"\nParamètres hand-eye sauvegardés dans '{OUTPUT_FILE}'")

robot.disconnect()
