import time
import json
import rerun as rr
import numpy as np
import pinocchio as pin
from pathlib import Path
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration

# ─── Configuration ────────────────────────────────────────────────────────────
URDF_PATH   = Path("/home/leo28/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
PACKAGE_DIR = Path("/home/leo28/SO-ARM100/Simulation/SO101")
PORT_LEADER   = "/dev/ttyACM0"
PORT_FOLLOWER = "/dev/ttyACM1"

CALIB_LEADER   = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_leader_arm.json"
CALIB_FOLLOWER = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower_arm.json"

MOTORS_LEADER = {
    "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
}

MOTORS_FOLLOWER = {
    "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ─── Charger la calibration depuis JSON ───────────────────────────────────────
def load_calibration(bus, calib_path):
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
    calibration = {}
    for motor_name, values in calib_data.items():
        calibration[motor_name] = MotorCalibration(
            id=values["id"],
            drive_mode=values["drive_mode"],
            homing_offset=values["homing_offset"],
            range_min=values["range_min"],
            range_max=values["range_max"],
        )
    bus.write_calibration(calibration)
    print(f"Calibration chargée depuis {calib_path.name}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def read_all(bus):
    positions = {}
    for name in JOINT_NAMES:
        try:
            positions[name] = bus.read("Present_Position", name, normalize=False)
        except Exception as e:
            positions[name] = 2048
    return positions

def write_all(bus, positions):
    for name in JOINT_NAMES:
        try:
            bus.write("Goal_Position", name, positions[name])
        except Exception as e:
            print(f"Erreur écriture {name}: {e}")

def ticks_to_rad(positions):
    q = np.zeros(len(JOINT_NAMES))
    for i, name in enumerate(JOINT_NAMES):
        ticks = positions[name]
        degrees = (ticks - 2048) * 360.0 / 4096.0
        q[i] = np.deg2rad(degrees)
    return q

def load_meshes(visual_model, prefix):
    for geom in visual_model.geometryObjects:
        mesh_path = PACKAGE_DIR / geom.meshPath
        if mesh_path.exists():
            rr.log(f"world/{prefix}/{geom.name}", rr.Asset3D(path=str(mesh_path)), static=True)

def update_viz(visual_model, visual_data, prefix):
    for geom_id, geom in enumerate(visual_model.geometryObjects):
        T = visual_data.oMg[geom_id]
        rr.log(f"world/{prefix}/{geom.name}", rr.Transform3D(translation=T.translation, mat3x3=T.rotation))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Charger URDF
    print("Chargement du modèle URDF...")
    model, _, visual_model = pin.buildModelsFromUrdf(str(URDF_PATH), package_dirs=[str(PACKAGE_DIR)])
    data        = model.createData()
    visual_data = pin.GeometryData(visual_model)
    print(f"Modèle chargé : {model.nq} DDL, {len(visual_model.geometryObjects)} pièces visuelles")

    # Connecter leader
    print(f"Connexion leader ({PORT_LEADER})...")
    bus_leader = FeetechMotorsBus(port=PORT_LEADER, motors=MOTORS_LEADER)
    bus_leader.connect()
    load_calibration(bus_leader, CALIB_LEADER)
    print("Leader connecté et calibré !")

    # Connecter follower
    print(f"Connexion follower ({PORT_FOLLOWER})...")
    bus_follower = FeetechMotorsBus(port=PORT_FOLLOWER, motors=MOTORS_FOLLOWER)
    bus_follower.connect()
    load_calibration(bus_follower, CALIB_FOLLOWER)
    print("Follower connecté et calibré !")

    # Initialiser rerun
    rr.init("so101_teleop_viz", spawn=True)
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Spatial3DView(origin="/", contents="/**"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Charger meshes follower
    print("Envoi des meshes 3D à rerun...")
    load_meshes(visual_model, prefix="follower")
    print("Meshes envoyés !")

    print("\nTéléopération + jumeau numérique lancés !")
    print("Bouge le leader — le follower suit et son jumeau s'affiche dans rerun.")
    print("Ctrl+C pour arrêter\n")

    try:
        while True:
            # Lire leader
            pos_leader = read_all(bus_leader)

            # Écrire follower (téléopération)
            write_all(bus_follower, pos_leader)

            # Lire position réelle follower
            pos_follower = read_all(bus_follower)

            # Cinématique + rerun
            q = ticks_to_rad(pos_follower)
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)
            update_viz(visual_model, visual_data, prefix="follower")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nArrêt propre...")
        bus_leader.disconnect()
        bus_follower.disconnect()
        print("Déconnecté.")

if __name__ == "__main__":
    main()