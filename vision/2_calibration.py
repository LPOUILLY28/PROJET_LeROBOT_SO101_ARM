"""
SCRIPT 2 - CALIBRATION DE LA CAMÉRA
Lance ce script après avoir capturé ~30 photos avec le script 1.
Il calcule les paramètres optiques de la caméra et les sauvegarde dans camera_params.npz
"""

import cv2
import numpy as np
import os
import glob

# ─── PARAMÈTRES ───────────────────────────────────────────────────────────────
PHOTOS_DIR = "photos_calibration"
CHESSBOARD_SIZE = (9, 6)       # coins internes : 9 colonnes, 6 lignes
CASE_SIZE_MM = 25.0            # ← MODIFIE CETTE VALEUR avec ta mesure réelle en mm
OUTPUT_FILE = "camera_params.npz"
# ──────────────────────────────────────────────────────────────────────────────

photos = sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.png")))
print(f"Photos trouvées : {len(photos)}")

if len(photos) < 10:
    print("Pas assez de photos (minimum 10). Lance d'abord le script 1_capture.py")
    exit()

# Points 3D théoriques du damier (dans le repère du damier)
# Ex: (0,0,0), (25,0,0), (50,0,0) ... en mm
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= CASE_SIZE_MM

objpoints = []  # points 3D réels
imgpoints = []  # points 2D dans l'image
valid_photos = []

print("\nAnalyse des photos...")
for path in photos:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_photos.append(path)
        print(f"  ✓ {os.path.basename(path)}")
    else:
        print(f"  ✗ {os.path.basename(path)} — damier non détecté, ignorée")

print(f"\nPhotos valides : {len(valid_photos)}/{len(photos)}")

if len(valid_photos) < 10:
    print("Pas assez de photos valides. Reprends la capture avec le script 1.")
    exit()

# Calibration
print("\nCalibration en cours...")
h, w = cv2.imread(valid_photos[0]).shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print(f"\nErreur de reprojection : {ret:.4f} px  (bon si < 1.0)")
print(f"\nMatrice intrinsèque K :")
print(f"  Focale x (fx) : {K[0,0]:.1f} px")
print(f"  Focale y (fy) : {K[1,1]:.1f} px")
print(f"  Centre optique : ({K[0,2]:.1f}, {K[1,2]:.1f}) px")
print(f"\nCoefficients de distorsion :")
print(f"  {dist.ravel()}")

# Sauvegarde
np.savez(OUTPUT_FILE, K=K, dist=dist, img_size=(w, h), case_size_mm=CASE_SIZE_MM)
print(f"\nParamètres sauvegardés dans '{OUTPUT_FILE}'")

# Visualisation de la distorsion sur une image
sample = cv2.imread(valid_photos[0])
undistorted = cv2.undistort(sample, K, dist)
cv2.imshow("Original", sample)
cv2.imshow("Corrigée (undistort)", undistorted)
print("\nAffichage : originale vs corrigée. Appuie sur une touche pour fermer.")
cv2.waitKey(0)
cv2.destroyAllWindows()
