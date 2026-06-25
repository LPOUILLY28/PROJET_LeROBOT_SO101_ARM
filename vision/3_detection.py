"""
SCRIPT 3 - DÉTECTION D'OBJET ET POSITION EN CM
Lance ce script après la calibration.
Il détecte un objet de couleur dans l'image et donne sa position (x, y) en cm
par rapport au centre de la zone de travail.
"""

import cv2
import numpy as np

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
PARAMS_FILE = "camera_params.npz"
CAMERA_HEIGHT_CM = 40.0    # ← hauteur de la caméra au-dessus de la table en cm (à mesurer)

# Couleur de l'objet à détecter en HSV
# Rouge par défaut — modifie ces valeurs si ton objet est d'une autre couleur
# Outil utile pour trouver les valeurs : https://github.com/harishtpj/ColorPicker
HSV_LOW  = np.array([0,   120, 70])   # borne basse
HSV_HIGH = np.array([10, 255, 255])   # borne haute
# Pour le bleu :  HSV_LOW=[100,120,70]  HSV_HIGH=[130,255,255]
# Pour le vert :  HSV_LOW=[40,50,50]    HSV_HIGH=[80,255,255]
# ──────────────────────────────────────────────────────────────────────────────

# Chargement des paramètres de calibration
try:
    data = np.load(PARAMS_FILE)
    K    = data["K"]
    dist = data["dist"]
    print("Calibration caméra chargée.")
except FileNotFoundError:
    print(f"'{PARAMS_FILE}' introuvable. Lance d'abord 2_calibration.py")
    exit()

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Détection en cours. Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire la caméra")
        break

    # Correction de la distorsion
    frame = cv2.undistort(frame, K, dist)

    # Conversion en HSV et masque de couleur
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    # Nettoyage morphologique (érosion puis dilatation)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Recherche du plus grand contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 300:
            M = cv2.moments(c)
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            # Conversion pixels → cm dans le repère caméra (vue de dessus)
            Z = CAMERA_HEIGHT_CM
            X_cm = (u - cx) * Z / fx
            Y_cm = (v - cy) * Z / fy

            # Affichage
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"Objet : X={X_cm:.1f}cm  Y={Y_cm:.1f}cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, "Q = quitter", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Detection objet - SO101", frame)
    cv2.imshow("Masque", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Détection terminée.")
