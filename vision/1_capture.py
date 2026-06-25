"""
SCRIPT 1 - CAPTURE DES PHOTOS DE CALIBRATION
Lance ce script, bouge le damier devant la caméra, appuie sur S pour sauvegarder.
Objectif : ~30 photos bien réparties sur toute la zone de travail.
"""

import cv2
import os
import numpy as np

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # index de l'iPhone (à adapter si besoin)
SAVE_DIR = "photos_calibration"
CHESSBOARD_SIZE = (9, 6)  # coins internes : 9 colonnes, 6 lignes
TARGET_PHOTOS = 30
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])
print(f"Caméra ouverte. Photos déjà sauvegardées : {count}/{TARGET_PHOTOS}")
print("Commandes : S = sauvegarder | Q = quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire la caméra")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cherche le damier dans l'image en temps réel
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        # Affine la détection au sous-pixel
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        status = "DAMIER DETECTE - appuie sur S pour sauvegarder"
        color = (0, 255, 0)
    else:
        status = "Damier non detecte - montre le damier entierement"
        color = (0, 0, 255)

    # Affichage des infos
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"Photos : {count}/{TARGET_PHOTOS}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Barre de progression
    progress = int((count / TARGET_PHOTOS) * frame.shape[1])
    cv2.rectangle(display, (0, frame.shape[0]-20), (progress, frame.shape[0]), (0, 255, 0), -1)

    cv2.imshow("Capture calibration - iPhone", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        if found:
            path = os.path.join(SAVE_DIR, f"photo_{count:03d}.png")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Sauvegarde : {path} ({count}/{TARGET_PHOTOS})")
            if count >= TARGET_PHOTOS:
                print("Objectif atteint ! Tu peux quitter avec Q.")
        else:
            print("  Damier non détecté, déplace-le et réessaie.")

cap.release()
cv2.destroyAllWindows()
print(f"\nCapture terminée : {count} photos sauvegardées dans '{SAVE_DIR}/'")
