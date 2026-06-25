"""
Script de simulation de la policy ACT dans MuJoCo pour le SO101.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import torch

# ─── Configuration ────────────────────────────────────────────────────────────
SCENE_PATH  = Path("/home/leo28/SO-ARM100/Simulation/SO101/scene.xml")
MODEL_PATH  = Path("/home/leo28/models/act_so101")
DEVICE      = "cpu"
N_STEPS     = 1000
DT          = 0.02  # 50 Hz

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

CAM_WIDTH  = 640
CAM_HEIGHT = 480

# ─── MuJoCo ──────────────────────────────────────────────────────────────────
def load_mujoco(scene_path):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data  = mujoco.MjData(model)
    return model, data

def get_joint_positions(model, data):
    positions = {}
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        positions[name] = float(data.qpos[joint_id]) if joint_id >= 0 else 0.0
    return positions

def set_joint_positions(model, data, actions):
    actions = np.atleast_1d(np.array(actions, dtype=float)).flatten()
    for i, name in enumerate(JOINT_NAMES):
        if i >= len(actions):
            break
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = actions[i]

def render_camera(model, data, renderer):
    renderer.update_scene(data)
    return renderer.render()

# ─── Policy ACT ───────────────────────────────────────────────────────────────
def load_policy(model_path, device="cpu"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    print(f"Chargement policy ACT depuis {model_path}...")
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy = policy.to(device)
    policy.eval()
    if hasattr(policy.config, 'temporal_ensemble_coeff'):
        policy.config.temporal_ensemble_coeff = None
    print("Policy ACT chargée !")
    return policy

def prepare_observation(joint_positions, image_rgb, device="cpu"):
    state = torch.tensor(
        [joint_positions[n] for n in JOINT_NAMES],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # (1, 6)

    img = torch.from_numpy(image_rgb).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    return {
        "observation.state": state,
        "observation.images.front": img,
    }

def predict_chunk(policy, model_mj, data, renderer, device="cpu"):
    """Prédit un chunk complet via predict_action_chunk."""
    joint_pos = get_joint_positions(model_mj, data)
    image     = render_camera(model_mj, data, renderer)
    obs       = prepare_observation(joint_pos, image, device=device)

    with torch.no_grad():
        # predict_action_chunk retourne (1, chunk_size, n_joints)
        actions = policy.predict_action_chunk(obs)

    actions = actions.squeeze(0).cpu().numpy()  # (chunk_size, n_joints)
    return actions

# ─── Boucle principale ────────────────────────────────────────────────────────
def run_simulation(policy, model, data, renderer, n_steps=N_STEPS, dt=DT):
    print(f"\nSimulation lancée — {n_steps} steps à {1/dt:.0f} Hz")
    print("Ferme la fenêtre MuJoCo pour arrêter.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step         = 0
        action_queue = []
        chunk_count  = 0

        while viewer.is_running() and step < n_steps:

            if len(action_queue) == 0:
                try:
                    chunk = predict_chunk(policy, model, data, renderer, device=DEVICE)
                    chunk = np.array(chunk, dtype=float)
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(1, -1)
                    action_queue = list(chunk)
                    chunk_count += 1
                    print(f"Chunk {chunk_count:3d} | {len(action_queue)} actions | range [{chunk.min():.3f}, {chunk.max():.3f}]")
                except Exception as e:
                    print(f"Erreur prédiction : {e}")
                    action_queue = [np.zeros(len(JOINT_NAMES))]

            action = action_queue.pop(0)
            set_joint_positions(model, data, action)

            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            time.sleep(dt)

        print(f"\nSimulation terminée — {step} steps, {chunk_count} chunks prédits.")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=== Simulation ACT sur SO101 dans MuJoCo ===\n")

    model, data = load_mujoco(SCENE_PATH)
    print(f"Scène chargée : {model.nq} DDL, {model.nu} actuateurs")

    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
    policy   = load_policy(MODEL_PATH, device=DEVICE)

    run_simulation(policy, model, data, renderer)

if __name__ == "__main__":
    main()
