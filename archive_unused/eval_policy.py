from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.factory import make_robot
import torch

# Charger la policy
print("Chargement de la policy...")
policy = ACTPolicy.from_pretrained("LEO2862/act_so101_pick_place")
policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)

# Connecter le robot
print("Connexion au robot...")
robot = make_robot(
    robot_type="so101_follower",
    robot_kwargs={
        "port": "/dev/ttyACM1",
        "cameras": {
            "front": {
                "type": "opencv",
                "index_or_path": "/dev/video2",
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }
    }
)

robot.connect()

# Boucle d'évaluation
num_episodes = 3
for episode in range(num_episodes):
    print(f"\n=== Épisode {episode + 1}/{num_episodes} ===")
    
    # Reset
    robot.teleop_to_starting_configuration()
    obs = robot.capture_observation()
    
    # Rollout
    for step in range(200):  # Max 200 steps par épisode
        # Préparer l'observation pour la policy
        obs_dict = {
            "observation.images.front": obs["observation.images.front"],
            "observation.state": obs["observation.state"]
        }
        
        # Inférence
        with torch.no_grad():
            action = policy.select_action(obs_dict)
        
        # Exécuter l'action
        obs, reward, done, info = robot.send_action(action)
        
        if done:
            print(f"Épisode terminé à l'étape {step}")
            break
    
    print(f"Succès: {info.get('success', False)}")

robot.disconnect()
print("\nÉvaluation terminée !")
