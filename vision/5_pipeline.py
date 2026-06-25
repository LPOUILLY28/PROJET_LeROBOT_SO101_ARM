"""
SCRIPT 5 - PIPELINE COMPLET PICK & PLACE
Détecte l'objet (cylindre blanc) avec l'iPhone, calcule sa position dans le
repère robot, fait la cinématique inverse, et commande le SO-101.

PRÉREQUIS :
- camera_params.npz  (script 2)
- handeye_params.npz (script 4)
- Robot connecté sur /dev/ttyACM0
- iPhone actif via Droidcam
"""

import cv2
import numpy as np
import time
from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig as SO101FollowerConfig

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
CAMERA_PARAMS     = "camera_params.npz"
HANDEYE_PARAMS    = "handeye_params.npz"
ROBOT_PORT        = "/dev/ttyACM0"
CAMERA_HEIGHT_CM  = 30.0    # hauteur iPhone au-dessus de la table

# Couleur de l'objet : cylindre blanc
# Le blanc en HSV : saturation faible, valeur élevée
HSV_LOW  = np.array([150, 100, 150])
HSV_HIGH = np.array([180, 255, 255])

# Hauteur de la table vue par le robot (Z de la pince quand elle attrape)
# À ajuster selon votre setup : hauteur de la pince au-dessus de la table en cm
Z_PICK_CM = 1.6   # pince 2cm au-dessus de la table pour attraper
Z_LIFT_CM = 15.0  # hauteur de sécurité pour les déplacements

# Longueurs des segments du SO-101 (depuis URDF officiel)
L1 = 6.24   # hauteur base → épaule (cm)
L2 = 10.13  # bras supérieur (cm)
L3 = 9.40   # avant-bras (cm)
L4 = 6.50   # outil/pince (cm)
# ──────────────────────────────────────────────────────────────────────────────

# ── Cinématique inverse SO-101 ──
def inverse_kinematics(x_cm, y_cm, z_cm):
    """
    Calcule les angles des servos pour atteindre (x, y, z) en cm
    dans le repère robot (base au centre).
    Retourne un dict d'angles en degrés, ou None si hors portée.
    
    Repère robot : x = devant, y = gauche, z = haut
    """
    # Angle de rotation de la base (shoulder_pan)
    shoulder_pan = np.degrees(np.arctan2(y_cm, x_cm))

    # Distance horizontale depuis la base
    r = np.sqrt(x_cm**2 + y_cm**2)

    # Hauteur à atteindre depuis l'épaule
    z_eff = z_cm - L1

    # Distance totale épaule → cible
    d = np.sqrt(r**2 + z_eff**2)

    # Vérification portée
    if d > (L2 + L3) or d < abs(L2 - L3):
        print(f"  Hors portée : distance={d:.1f}cm, max={L2+L3:.1f}cm")
        return None

    # Loi des cosinus pour elbow_flex
    cos_elbow = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
    cos_elbow = np.clip(cos_elbow, -1, 1)
    elbow_flex = np.degrees(np.arccos(cos_elbow))

    # Angle épaule
    alpha = np.degrees(np.arctan2(z_eff, r))
    cos_beta = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
    cos_beta = np.clip(cos_beta, -1, 1)
    beta = np.degrees(np.arccos(cos_beta))
    shoulder_lift = -(alpha + beta)

    # Poignet : garder la pince vers le bas
    wrist_flex = -(shoulder_lift + elbow_flex - 90)

    return {
        'shoulder_pan':  shoulder_pan,
        'shoulder_lift': shoulder_lift,
        'elbow_flex':    elbow_flex,
        'wrist_flex':    wrist_flex,
        'wrist_roll':    0.0,
        'gripper':       10,   # ouvert
    }

def send_pose(robot, pose, gripper_val=None):
    """Envoie une consigne de position au robot."""
    if gripper_val is not None:
        pose = dict(pose)
        pose['gripper'] = gripper_val
    action = {f"{k}.pos": v for k, v in pose.items()}
    robot.send_action(action)

def detect_object(frame, K, dist):
    """Détecte le cylindre blanc et retourne (u, v) en pixels."""
    frame_ud = cv2.undistort(frame, K, dist)
    hsv = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame_ud, mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return None, frame_ud, mask

    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    cv2.drawContours(frame_ud, [c], -1, (0,255,0), 2)
    cv2.circle(frame_ud, (u,v), 6, (0,255,0), -1)
    cv2.putText(frame_ud, "Objet detecte", (u+10, v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return (u, v), frame_ud, mask

# ── Chargement paramètres ──
try:
    cam_data = np.load(CAMERA_PARAMS)
    K    = cam_data["K"]
    dist = cam_data["dist"]
    print("Calibration caméra chargée.")
except FileNotFoundError:
    print(f"'{CAMERA_PARAMS}' introuvable. Lance d'abord 2_calibration.py"); exit()

try:
    he_data = np.load(HANDEYE_PARAMS)
    R_he = he_data["R"]
    t_he = he_data["t"]
    print("Calibration hand-eye chargée.")
except FileNotFoundError:
    print(f"'{HANDEYE_PARAMS}' introuvable. Lance d'abord 4_handeye.py"); exit()

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

# ── Connexion robot ──
print(f"Connexion robot sur {ROBOT_PORT}...")
config = SO101FollowerConfig(port=ROBOT_PORT)
robot  = SO101Follower(config)
robot.connect()
print("Robot connecté.\n")

# Position home
HOME = {'shoulder_pan': 0, 'shoulder_lift': -90, 'elbow_flex': 90,
        'wrist_flex': 0, 'wrist_roll': 0, 'gripper': 10}
send_pose(robot, HOME)
time.sleep(2.0)

# ── Caméra ──
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Détection en cours. Appuie sur ESPACE pour lancer le pick, Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    uv, display, mask = detect_object(frame, K, dist)

    if uv:
        u, v_px = uv

        # pixels → repère caméra (cm)
        Z = CAMERA_HEIGHT_CM
        X_cam = (u    - cx) * Z / fx
        Y_cam = (v_px - cy) * Z / fy

        # repère caméra → repère robot (cm)
        pos_cam   = np.array([X_cam, Y_cam, Z])
        pos_robot = R_he @ pos_cam + t_he

        x_r, y_r = pos_robot[0], pos_robot[1]

        pos_text = f"Robot: X={x_r:.1f}cm  Y={y_r:.1f}cm"
        cv2.putText(display, pos_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(display, "ESPACE = pick & place", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    cv2.putText(display, "Q = quitter", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.imshow("Pick & Place SO-101", display)
    cv2.imshow("Masque objet", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' ') and uv:
        print(f"\n--- PICK & PLACE ---")
        print(f"Objet détecté : X={x_r:.1f}cm  Y={y_r:.1f}cm")

        # 1. Cinématique inverse pour la position de pick
        pose_pick = inverse_kinematics(x_r, y_r, Z_PICK_CM)
        if pose_pick is None:
            print("Objet hors portée du robot !")
            continue

        # 2. Position intermédiaire en hauteur (évite de raser la table)
        pose_above = dict(pose_pick)
        pose_above_ik = inverse_kinematics(x_r, y_r, Z_LIFT_CM)
        if pose_above_ik is None:
            print("Position de levée hors portée !")
            continue

        print("1. Ouverture pince...")
        send_pose(robot, HOME, gripper_val=10)
        time.sleep(1.0)

        print("2. Descente vers l'objet...")
        send_pose(robot, pose_above_ik, gripper_val=10)
        time.sleep(2.0)
        send_pose(robot, pose_pick, gripper_val=10)
        time.sleep(2.0)

        print("3. Fermeture pince (saisie)...")
        send_pose(robot, pose_pick, gripper_val=50)
        time.sleep(1.5)

        print("4. Levée de l'objet...")
        send_pose(robot, pose_above_ik, gripper_val=50)
        time.sleep(2.0)

        print("5. Retour home avec objet...")
        send_pose(robot, HOME, gripper_val=50)
        time.sleep(2.0)

        print("6. Dépôt de l'objet...")
        send_pose(robot, HOME, gripper_val=10)
        time.sleep(1.0)

        print("Pick & place terminé !\n")

cap.release()
cv2.destroyAllWindows()
send_pose(robot, HOME)
time.sleep(1.0)
robot.disconnect()
print("Robot déconnecté.")
