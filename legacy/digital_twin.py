import time
import rerun as rr
import numpy as np
import pinocchio as pin
from pathlib import Path
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

URDF_PATH   = Path("/home/leo28/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
PACKAGE_DIR = Path("/home/leo28/SO-ARM100/Simulation/SO101")
PORT        = "/dev/ttyACM1"

# Résolution encodeur STS3215 : 4096 ticks = 360°
TICKS_PER_REV  = 4096
CENTER_TICK    = 2048
MAX_DEGREES    = 180.0   # plage max autorisée par joint (±180°)

MOTORS = {
    "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    "gripper":       Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Signe par joint : +1 = sens normal, -1 = inverser si le visuel part dans l'autre sens
# Ajuste ces valeurs si un joint est miroir dans rerun
JOINT_SIGNS = {
    "shoulder_pan":  1,
    "shoulder_lift": 1,
    "elbow_flex":    1,
    "wrist_flex":    1,
    "wrist_roll":    1,
    "gripper":       1,
}

# Offset en degrés par joint si la position "neutre" du moteur ≠ 2048
# Laisse à 0 pour commencer, ajuste si un joint a un biais visible
JOINT_OFFSETS_DEG = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}

def ticks_to_rad(ticks, name):
    """Convertit des ticks encodeur bruts (0-4095) en radians,
    centré sur 2048, avec signe et offset par joint."""
    degrees = (ticks - CENTER_TICK) * 360.0 / TICKS_PER_REV
    degrees += JOINT_OFFSETS_DEG[name]
    degrees *= JOINT_SIGNS[name]
    # Clamp pour éviter des valeurs aberrantes
    degrees = np.clip(degrees, -MAX_DEGREES, MAX_DEGREES)
    return np.deg2rad(degrees)

def read_all_positions(bus):
    positions = {}
    for name in JOINT_NAMES:
        try:
            val = bus.read("Present_Position", name, normalize=False)
            # Sécurité : valeur hors plage → on garde le centre
            if not (0 <= val <= 4095):
                print(f"[WARN] {name}: tick hors plage ({val}), remplacé par 2048")
                val = CENTER_TICK
            positions[name] = val
        except Exception as e:
            print(f"Erreur lecture {name}: {e}")
            positions[name] = CENTER_TICK
    return positions

def main():
    print("Chargement du modèle URDF + meshes...")
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(URDF_PATH),
        package_dirs=[str(PACKAGE_DIR)]
    )
    data        = model.createData()
    visual_data = pin.GeometryData(visual_model)
    print(f"Modèle chargé : {model.nq} degrés de liberté")
    print(f"Pièces visuelles : {len(visual_model.geometryObjects)}")

    # Position neutre (tous joints à 0 rad) pour initialiser
    q_neutral = pin.neutral(model)
    print(f"Position neutre URDF : {np.round(np.rad2deg(q_neutral), 1)} °")

    print(f"Connexion au bus sur {PORT}...")
    bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
    bus.connect()
    print("Bus connecté !")

    # Initialiser rerun
    rr.init("so101_digital_twin", spawn=True)
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Spatial3DView(origin="/", contents="/**"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Charger les meshes STL une seule fois (static=True → envoyé une fois)
    print("Envoi des meshes 3D à rerun...")
    missing = 0
    for geom in visual_model.geometryObjects:
        mesh_path = PACKAGE_DIR / geom.meshPath
        if mesh_path.exists():
            rr.log(
                f"world/robot/{geom.name}",
                rr.Asset3D(path=str(mesh_path)),
                static=True,
            )
        else:
            print(f"[WARN] Mesh introuvable : {mesh_path}")
            missing += 1
    print(f"Meshes envoyés ! ({missing} manquants)")

    print("Jumeau numérique lancé — bouge le bras !")
    print("Ctrl+C pour arrêter\n")

    try:
        while True:
            positions = read_all_positions(bus)

            # Construire le vecteur de configuration q
            # On part de la position neutre URDF pour respecter les offsets du modèle
            q = q_neutral.copy()
            for i, name in enumerate(JOINT_NAMES):
                if i < model.nq:
                    q[i] = ticks_to_rad(positions[name], name)

            # Debug optionnel : affiche les angles toutes les 2 secondes
            if int(time.time() * 10) % 20 == 0:
                angles_deg = [round(np.rad2deg(q[i]), 1) for i in range(model.nq)]
                print(f"q (°): {dict(zip(JOINT_NAMES, angles_deg))}")

            # Cinématique directe
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)

            # Envoyer les transforms de chaque pièce à rerun
            for geom_id, geom in enumerate(visual_model.geometryObjects):
                T = visual_data.oMg[geom_id]
                rr.log(
                    f"world/robot/{geom.name}",
                    rr.Transform3D(
                        translation=T.translation,
                        mat3x3=T.rotation,
                    )
                )

            time.sleep(0.05)  # ~20 Hz

    except KeyboardInterrupt:
        print("\nArrêt du jumeau numérique")
        bus.disconnect()
        print("Déconnecté proprement")

if __name__ == "__main__":
    main()