"""
teleop_mujoco.py — version basse latence
════════════════════════════════════════════════════════════
Téleopération temps réel : Leader physique SO101 → MuJoCo

Optimisations :
  • sync_read     : lecture groupée des 6 servos en 1 paquet (~3-5× plus rapide)
  • thread reader : la lecture USB ne bloque jamais le viewer
  • filtre léger  : ALPHA réduit pour moins d'inertie

Usage :
  python teleop_mujoco.py                    # auto-détection
  python teleop_mujoco.py --port /dev/ttyACM1
  python teleop_mujoco.py --scene chemin.xml
  python teleop_mujoco.py --simulate         # sans leader
════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# ─── Configuration ────────────────────────────────────────────────────────────
SCENE_CANDIDATES = [
    Path("/home/leo28/SO-ARM100/Simulation/SO101/scene_pick_place.xml"),
    Path.home() / "SO-ARM100/Simulation/SO101/scene_pick_place.xml",
    Path.cwd() / "scene_pick_place.xml",
]
SCENE_FALLBACK = Path("/home/leo28/SO-ARM100/Simulation/SO101/scene.xml")

LEADER_PORT = "/dev/ttyACM2"
CALIB_PATH  = Path("/home/leo28/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_leader_arm.json")

FREQ_HZ      = 100        # viewer plus réactif
DT           = 1.0 / FREQ_HZ
ALPHA_FILTER = 0.15       # moins d'inertie

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]


# ─── MuJoCo ───────────────────────────────────────────────────────────────────
def load_mujoco(scene_path):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data  = mujoco.MjData(model)
    return model, data


def set_joint_positions(model, data, actions):
    actions = np.atleast_1d(np.array(actions, dtype=float)).flatten()
    for i, name in enumerate(JOINT_NAMES):
        if i >= len(actions):
            break
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            data.ctrl[aid] = actions[i]


# ─── LeRobot ──────────────────────────────────────────────────────────────────
def _charger_lerobot():
    try:
        import serial.tools.list_ports
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration

        motors = {name: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
                  for i, name in enumerate(JOINT_NAMES)}
        return serial, FeetechMotorsBus, motors, MotorCalibration, True
    except ImportError as e:
        print(f"[WARN] LeRobot non disponible : {e}")
        return None, None, None, None, False


def detecter_port_leader(serial_module, preferer=LEADER_PORT):
    ports = list(serial_module.tools.list_ports.comports())
    devices = [p.device for p in ports]
    if preferer and preferer in devices:
        print(f"[Leader] Port préféré trouvé : {preferer}")
        return preferer
    for p in ports:
        if "ttyACM" in p.device or "ttyUSB" in p.device:
            print(f"[Leader] Port détecté : {p.device}")
            return p.device
    return None


# ─── Bras leader avec thread reader ──────────────────────────────────────────
class LeaderArm:
    def __init__(self, port, simulate=False):
        serial_mod, FeetechMotorsBus, motors, MotorCalibration, lerobot_ok = _charger_lerobot()

        self.simulate = simulate or not lerobot_ok or port is None
        self._bus = None
        self._last = np.zeros(len(JOINT_NAMES))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._use_sync_read = False
        self._warned = False

        if self.simulate:
            print("[Leader] Mode STATIQUE — pas de leader, bras au repos.")
            return

        try:
            self._bus = FeetechMotorsBus(port=port, motors=motors)
            self._bus.connect()

            if CALIB_PATH.exists():
                with open(CALIB_PATH) as f:
                    raw = json.load(f)
                calib = {
                    name: MotorCalibration(
                        id=int(p["id"]),
                        drive_mode=int(p["drive_mode"]),
                        homing_offset=int(p["homing_offset"]),
                        range_min=int(p["range_min"]),
                        range_max=int(p["range_max"]),
                    )
                    for name, p in raw.items()
                }
                self._bus.calibration = calib
                if hasattr(self._bus, "write_calibration"):
                    self._bus.write_calibration(calib)
                print(f"[Leader] Calibration chargée et écrite ({CALIB_PATH.name})")
            else:
                print(f"[WARN] Calibration introuvable : {CALIB_PATH}")

            try:
                self._bus.disable_torque()
                print("[Leader] Torque désactivé.")
            except Exception as e:
                print(f"[Leader] (info) disable_torque indisponible : {e}")

            self._use_sync_read = hasattr(self._bus, "sync_read")
            mode = "sync_read (groupé)" if self._use_sync_read else "read séquentiel"
            print(f"[Leader] Mode lecture : {mode}")

            print(f"[Leader] Connecté sur {port}")
            print("[Leader] Bouge le bras — le jumeau MuJoCo va suivre !\n")

            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"[Leader] Erreur connexion : {e}")
            self.simulate = True
            self._bus = None

    def _read_once(self):
        try:
            if self._use_sync_read:
                vals = self._bus.sync_read("Present_Position")
                angles_deg = np.array([float(vals[n]) for n in JOINT_NAMES])
            else:
                angles_deg = np.array([
                    float(self._bus.read("Present_Position", n)) for n in JOINT_NAMES
                ])
            return np.deg2rad(angles_deg)
        except Exception as e:
            if not self._warned:
                print(f"[Leader] Erreur lecture : {e}")
                self._warned = True
            return None

    def _reader_loop(self):
        filt = None
        while not self._stop_event.is_set():
            angles = self._read_once()
            if angles is not None:
                if filt is None:
                    filt = angles
                else:
                    filt = ALPHA_FILTER * filt + (1 - ALPHA_FILTER) * angles
                with self._lock:
                    self._last = filt

    def lire_positions_rad(self):
        with self._lock:
            return self._last.copy()

    def deconnecter(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._bus is not None:
            try:
                self._bus.disconnect()
                print("[Leader] Déconnecté.")
            except Exception:
                pass


# ─── Boucle principale ────────────────────────────────────────────────────────
def run(scene_path, leader):
    print(f"[MuJoCo] Chargement de la scène : {scene_path}")
    model, data = load_mujoco(scene_path)
    print(f"[MuJoCo] Modèle : {model.nq} DDL, {model.nu} actuateurs\n")

    for nom in JOINT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nom)
        status = "✓" if aid >= 0 else "✗ INTROUVABLE"
        print(f"  Actuateur '{nom}' → id={aid}  {status}")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.monotonic()
            set_joint_positions(model, data, leader.lire_positions_rad())
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def trouver_scene(scene_arg):
    if scene_arg:
        p = Path(scene_arg).expanduser()
        return p if p.exists() else None
    for p in SCENE_CANDIDATES:
        if p.exists():
            print(f"[Scene] Trouvée : {p}")
            return p
    if SCENE_FALLBACK.exists():
        print(f"[WARN] scene_pick_place.xml introuvable → fallback scene.xml")
        return SCENE_FALLBACK
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--scene", default=None, help="Chemin vers un .xml MuJoCo")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    scene_path = trouver_scene(args.scene)
    if scene_path is None:
        print("[ERREUR] Aucune scène MuJoCo trouvée.")
        return

    port = None
    if not args.simulate:
        serial_mod, _, _, _, lerobot_ok = _charger_lerobot()
        if lerobot_ok:
            port = args.port or detecter_port_leader(serial_mod)
        if port is None:
            print("[WARN] Leader non trouvé.")
            rep = input("  Continuer en mode statique ? [o/n] : ").strip().lower()
            if rep != "o":
                return
            args.simulate = True

    leader = LeaderArm(port=port, simulate=args.simulate)
    try:
        run(scene_path, leader)
    except KeyboardInterrupt:
        print("\n[Info] Arrêt demandé.")
    finally:
        leader.deconnecter()
        print("Terminé.")


if __name__ == "__main__":
    main()
