import argparse
import logging
import sys
from matplotlib import pyplot as plt
import pybullet as p
import time
import pybullet_data
import numpy as np
from agent import Agent
from config import TARGET_POSITION, BATCH_SIZE, GAMMA, GUI_MODE, ACTOR_LR, CRITIC_LR, MAX_ACTION_VALUE, MAX_RM_SIZE, MIN_ACTION_VALUE, TAU, TIMESTEPS_PER_EPOCH, TOTAL_EPOCHS, WARMUP_TIMESTEPS
import os
from utils import get_date_time

# Parse CLI args
parser = argparse.ArgumentParser()

parser.add_argument(
    "--note", help="Quick run test note", type=str, default="")
# parser.add_argument(
#     "--test_id", help="Test ID to run", type=int, default=0)
parser.add_argument(
    "--timesteps", help="Total timesteps per epoch", type=int, default=TIMESTEPS_PER_EPOCH)
parser.add_argument(
    "--epochs", help="Total epochs to run", type=int, default=TOTAL_EPOCHS)
parser.add_argument(
    "--warmup", help="Warmup steps to run", type=int, default=WARMUP_TIMESTEPS)
parser.add_argument(
    "--memory_size", help="Replay memory max size", type=int, default=MAX_RM_SIZE)
parser.add_argument(
    "--batch_size", help="Memory batch size for learning", type=int, default=BATCH_SIZE)
parser.add_argument(
    "--actor_lr", help="Actor neural network learning rate", type=float, default=ACTOR_LR)
parser.add_argument(
    "--critic_lr", help="Critic neural network learning rate", type=float, default=CRITIC_LR)
parser.add_argument(
    "--gamma", help="Gamma", type=float, default=GAMMA)
parser.add_argument(
    "--tau", help="Polyak averaging coefficient", type=float, default=TAU)
parser.add_argument(
    "--target_position", help="Target position (ex: [20, 20])", type=list, default=TARGET_POSITION)
parser.add_argument(
    "--gui", help="Use GUI mode", type=bool, default=GUI_MODE)

parser.add_argument("--algorithm", choices=["sac", "td3"],
                    help="Actor-Critic algorithm", type=str, default="sac")

cli_args = parser.parse_args()

# Run test parameters
note = cli_args.note
# test_id = cli_args.test_id
timesteps = cli_args.timesteps
epochs = cli_args.epochs
warmup = cli_args.warmup
memory_size = cli_args.memory_size
batch_size = cli_args.batch_size
actor_lr = cli_args.actor_lr
critic_lr = cli_args.critic_lr
gamma = cli_args.gamma
tau = cli_args.tau
target_position = cli_args.target_position
gui = cli_args.gui
algorithm = cli_args.algorithm

# Run test folder
runtest_folder_name = f"run_test_{get_date_time()}"

# Create folders if not created
if not os.path.isdir(runtest_folder_name):
    os.mkdir(runtest_folder_name)

if not os.path.isdir(f"{runtest_folder_name}/logs"):
    os.mkdir(f"{runtest_folder_name}/logs")

if not os.path.isdir(f"{runtest_folder_name}/checkpoints"):
    os.mkdir(f"{runtest_folder_name}/checkpoints")

if not os.path.isdir(f"{runtest_folder_name}/plots"):
    os.mkdir(f"{runtest_folder_name}/plots")

with open(f"{runtest_folder_name}/run_test_observations.txt", "w") as f:
    f.write(note)

with open(f"{runtest_folder_name}/run_test_config.txt", "w") as f:
    f.write(f"TIMESTEPS PER EPOCH: {timesteps}\n")
    f.write(f"TOTAL NUMBER OF EPOCHS: {epochs}\n")
    f.write(f"WARMUP STEPS: {warmup}\n")
    f.write(f"MEMORY SIZE: {memory_size}\n")
    f.write(f"BATCH SIZE: {batch_size}\n")
    f.write(f"ACTOR LEARNING RATE: {actor_lr}\n")
    f.write(f"CRITIC LEARNING RATE: {critic_lr}\n")
    f.write(f"GAMMA: {gamma}\n")
    f.write(f"POLYAK AVERAGING COEFFICIENT: {tau}\n")
    f.write(f"TARGET DESTINATION POSITION: {target_position}\n")
    f.write(f"Algorithm: {algorithm}")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    filename=f"{runtest_folder_name}/logs/epochs.log")

# or p.DIRECT for non-graphical version
physicsClientId = p.connect(p.GUI if gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.8)
# Spawn ground
plane_id = p.loadURDF("plane.urdf")
# Robot spawn position and orientation
cubeStartPos = [0, 0, 1.3]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 1])
# Spawn robot
robot_id = p.loadURDF("./robot.urdf",
                      cubeStartPos, cubeStartOrientation, useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
# Spawn target
tableId = p.loadURDF("table/table.urdf", basePosition=[
                     TARGET_POSITION[0], TARGET_POSITION[1], 0], physicsClientId=physicsClientId)

# Score history and best epoch score
score_history = []
best_score = float("-inf")


# Joints info
num_joints = p.getNumJoints(robot_id)
joint_ids = [i for i in range(num_joints)]
joints_max_limit = []
joints_min_limit = []


MAX_TORQUE_FORCE = 750
OBSERVATION_SPACE_DIM = 50
ACTION_SPACE_DIM = num_joints
last_action = [0 for _ in range(ACTION_SPACE_DIM)]
print(f"""Robot num joints: {num_joints} | State space dim: {
      OBSERVATION_SPACE_DIM} | Action space dim: {ACTION_SPACE_DIM}""")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)

    name = joint_info[1]
    type = joint_info[2]
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    max_force = joint_info[10]
    max_velocity = joint_info[11]

    joints_min_limit.append(lower_limit)
    joints_max_limit.append(upper_limit)


# Create agent
agent = Agent(start_steps=warmup, action_space_dims=ACTION_SPACE_DIM,
              state_space_dims=OBSERVATION_SPACE_DIM, actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, max_action=MAX_ACTION_VALUE, min_action=MIN_ACTION_VALUE, max_replay_size=memory_size, tau=tau, alpha=0.2, runtest_folder_name=runtest_folder_name)


def reset_env():
    # p.setJointMotorControlArray(robot_id, [i for i in range(
    #     num_joints)], p.POSITION_CONTROL, targetPositions=[0 for _ in range(num_joints)])
    p.resetBasePositionAndOrientation(
        robot_id, cubeStartPos, cubeStartOrientation)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, targetValue=0, targetVelocity=0)
    update_last_ts_robot_on_ground_flag(False)


def scale_value(x, lower_limit, upper_limit):
    # return (abs(upper_limit)-abs(lower_limit)) * x + abs(lower_limit)
    return ((x+1)/2)*(upper_limit-lower_limit)+lower_limit


def rescale_actions(nn_actions):
    target_actions = []
    for current_action, lower_limit, upper_limit in list(zip(nn_actions, joints_min_limit, joints_max_limit)):
        target_actions.append(
            scale_value(current_action, lower_limit, upper_limit)
        )

    return target_actions


def is_state_success():
    (x, y, _), _ = p.getBasePositionAndOrientation(robot_id)
    X_TARGET, Y_TARGET = TARGET_POSITION
    return (abs(x - X_TARGET) < 0.05 and abs(y-Y_TARGET) < 0.05)


# def get_relative_distance_from_target(current_x, current_y):
#     X_TARGET, Y_TARGET = TARGET_POSITION

#     return [abs(X_TARGET-current_x), abs(Y_TARGET-current_y)]


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# TODO: Big base robot
# def get_contact_sensor_values():
#     heel_1_contact = min(len(p.getContactPoints(
#         robot_id, plane_id, linkIndexA=3)), 1)
#     foot_1_contact = min(len(p.getContactPoints(
#         robot_id, plane_id, linkIndexA=4)), 1)

#     heel_2_contact = min(len(p.getContactPoints(
#         robot_id, plane_id, linkIndexA=8)), 1)
#     foot_2_contact = min(len(p.getContactPoints(
#         robot_id, plane_id, linkIndexA=9)), 1)

#     return [
#         max(foot_1_contact, heel_1_contact),
#         max(foot_2_contact, heel_2_contact)
#     ]
# TODO: SMALL BASE ROBOT


def get_contact_sensor_values():
    foot_1_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=4)), 1)

    foot_2_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=9)), 1)

    return [
        foot_1_contact,
        foot_2_contact
    ]


def is_state_truncated(timestep):
    return timestep == timesteps-1


def get_distance_from_target_diff(state, new_state):
    # past_state_distance_from_target = np.square(
    #     state[0] - TARGET_POSITION[0]) + np.square(state[1] - TARGET_POSITION[1])
    past_state_distance_from_target = np.linalg.norm(
        np.array(state[:2]) - np.array(TARGET_POSITION[:2])
    )

    # new_state_distance_from_target = np.square(
    #     new_state[0] - TARGET_POSITION[0]) + np.square(new_state[1] - TARGET_POSITION[1])
    new_state_distance_from_target = np.linalg.norm(
        np.array(new_state[:2]) - np.array(TARGET_POSITION[:2])
    )

    distance_from_targets_diff = past_state_distance_from_target - \
        new_state_distance_from_target

    return distance_from_targets_diff


def get_center_of_mass():
    dynamics_info = p.getDynamicsInfo(robot_id, -1)
    # Center of mass position in local coordinates
    com_position = dynamics_info[3]
    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)

    com_world = p.multiplyTransforms(
        base_position, base_orientation, com_position, (0, 0, 0, 1)
    )[0]

    return com_world


def get_current_observation():
    observation = np.array([])

    for joint_id in range(num_joints):
        # Append joint rotation and velocity to observation (10 motors * 2 states = 20 items)
        joint_rotation, joint_velocity, _, _ = p.getJointState(
            robot_id, joint_id)
        observation = np.append(observation, [joint_rotation, joint_velocity])

    # Append robot linear and angular velocity to observation (2 velocities * 3 dimensions = 6 items)
    robot_linear_velocity, robot_angular_velocity = p.getBaseVelocity(
        robot_id)
    observation = np.append(
        observation, [robot_linear_velocity, robot_angular_velocity])

    # Append current x, y coordinates and height (z)
    (current_x, current_y, height), orientation = p.getBasePositionAndOrientation(robot_id)
    observation = np.append(observation, [current_x, current_y, height])

    # Append euclidian distance from target (1 item)
    observation = np.append(
        observation,
        euclidean_distance([current_x, current_y], TARGET_POSITION)
    )

    # Append center of mass
    observation = np.append(observation, get_center_of_mass())

    # Append orientation
    observation = np.append(observation, orientation)

    # Append contact sensors (binary array with two elements: feet touching the ground) (2 items)
    observation = np.append(observation, get_contact_sensor_values())

    # Append robot fell state (1 item)
    observation = np.append(observation, get_robot_fell_state())

    # Append last action
    observation = np.append(observation, last_action)

    # Total of 101 elements
    return observation


def is_robot_on_ground():
    return len(p.getContactPoints(robot_id, plane_id, linkIndexA=-1)) > 0


# Indicates if the robot, in the last timestep, was on the ground
last_ts_robot_on_ground_flag = False


def update_last_ts_robot_on_ground_flag(flag: bool):
    global last_ts_robot_on_ground_flag
    last_ts_robot_on_ground_flag = flag


def get_robot_fell_state():
    return last_ts_robot_on_ground_flag


def is_robot_up() -> bool:
    (_, _, base_height), _ = p.getBasePositionAndOrientation(robot_id)
    return base_height > 0.8


def is_robot_on_ground() -> bool:
    (_, _, base_height), _ = p.getBasePositionAndOrientation(robot_id)
    return base_height < 0.4


def robot_fell():
    return (not last_ts_robot_on_ground_flag and is_robot_on_ground())


def recovered_from_fall():
    return (last_ts_robot_on_ground_flag and is_robot_up())


def can_plot_rewards(current_epoch) -> bool:
    return current_epoch % 50 == 0


def making_progress_towards_getting_up(past_position, new_position):
    past_height = past_position[2]
    new_height = new_position[2]
    return new_height - past_height > 0.01


rewards_data_plot = []


def get_reward(past_position, new_position, epoch):
    timestep_reward = 0

    robot_fell_penalty = -15 if robot_fell() else 0
    robot_recovered_reward = 20 if recovered_from_fall() else 0

    timestep_reward += robot_fell_penalty
    timestep_reward += robot_recovered_reward

    if is_robot_up():
        update_last_ts_robot_on_ground_flag(False)
    elif is_robot_on_ground():
        update_last_ts_robot_on_ground_flag(True)
    # Else doesnt update

    global_robot_fell_state = get_robot_fell_state()

    # Get torque applied to each joint
    # torque_applied = 0
    # for i in joint_ids:
    #     torque_applied += abs(p.getJointState(robot_id, i)[3])

    # force_penalty = -(np.mean(torque_applied))*0.01
    # timestep_reward += force_penalty

    # print(f"force_penalty: {force_penalty}")
    # print("-"*50)

    (x_speed, y_speed, _), _ = p.getBaseVelocity(robot_id)

    # speed = 0
    # if global_robot_fell_state:
    #     speed = np.sqrt(np.square(x_speed)+np.square(y_speed)) * 1e-2
    # elif not global_robot_fell_state:
    speed = np.sqrt(np.square(x_speed)+np.square(y_speed)) * 1e-1
    timestep_reward += speed

    robot_on_ground_continuous_penalty = -1 if global_robot_fell_state else 1

    timestep_reward += robot_on_ground_continuous_penalty

    making_progress_from_fall = 0

    if global_robot_fell_state and making_progress_towards_getting_up(past_position, new_position):
        making_progress_from_fall = 5  # Reward for partial progress at getting up\

    timestep_reward += making_progress_from_fall

    both_feet_touching_ground_reward = 0

    foot_contact_readings = get_contact_sensor_values()
    # If both feet touching the ground, receive a reward
    if not global_robot_fell_state and foot_contact_readings[0] and foot_contact_readings[1]:
        both_feet_touching_ground_reward = 0.5

    timestep_reward += both_feet_touching_ground_reward

    if is_state_success():
        timestep_reward += 300

    relative_distance_from_target = 0

    if not global_robot_fell_state:
        relative_distance_from_target = get_distance_from_target_diff(
            past_position, new_position) * 1e2

    timestep_reward += relative_distance_from_target

    if can_plot_rewards(epoch):
        rewards_data_plot.append({
            # "force_penalty": force_penalty,
            "speed": speed,
            "robot_fell_penalty": robot_fell_penalty,
            "robot_recovered_reward": robot_recovered_reward,
            "robot_on_ground_continuous_penalty": robot_on_ground_continuous_penalty,
            "relative_distance_from_target": relative_distance_from_target,
            "making_progress_towards_getting_up": making_progress_from_fall,
            "foot_contact_readings": both_feet_touching_ground_reward,
            "success_reward": 300 if is_state_success() else 0,
            "total_reward": timestep_reward
        })

    return timestep_reward


def log_divider():
    logger.info("-"*50)


for epoch in range(TOTAL_EPOCHS):
    epoch_start_timestamp = get_date_time()
    epoch_acc_score = 0

    # log_divider()
    logger.info(f"EPOCH #{epoch}/{TOTAL_EPOCHS}:")
    logger.info(f"\t- Status:\tSTARTED")
    logger.info(f"\t- Start date:\t{epoch_start_timestamp}")
    log_divider()

    for timestep in range(timesteps):
        position, _ = p.getBasePositionAndOrientation(robot_id)
        p.resetDebugVisualizerCamera(3, 45, -25, position)

        # Get current observation
        current_observation = get_current_observation()

        # Decide whether or not to add noise
        add_noise = False

        # Get actions | range: [-1, 1]
        actions = agent.select_actions(current_observation, add_noise)
        last_action = actions

        rescaled_position_actions = rescale_actions(actions)

        # print("Rescaled position actions:")
        # for a in rescaled_position_actions:
        #     print(f"\t - {a}")

        # print("Rescaled force actions:")
        # for a in rescaled_force_actions:
        #     print(f"\t - {a}")

        # print("-."*50)

        # Apply actions
        for i in joint_ids:
            p.setJointMotorControl2(robot_id, i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=rescaled_position_actions[i], maxVelocity=100, force=MAX_TORQUE_FORCE)

        # Perform actions in the environment
        p.stepSimulation()

        if gui:
            time.sleep(1./240.)

        # Get new observation
        new_observation = get_current_observation()

        # Get new robot positions
        new_position, _ = p.getBasePositionAndOrientation(robot_id)

        # Get timestep reward
        reward = get_reward(position, new_position, epoch)

        # Add reward to epoch accumulated score
        epoch_acc_score += reward

        # Check if epoch truncated
        terminal_flag = robot_fell()

        agent.remember(current_observation, actions, reward,
                       new_observation, terminal_flag)

        agent.learn()

        if is_state_success():
            break

    print("FIM")

    if can_plot_rewards(epoch):
        x = [i for i in range(len(rewards_data_plot))]
        plt.figure(figsize=(12, 8), dpi=200)
        plt.title(f"Rewards plot from epoch {epoch}")
        # plt.plot(x, [data["force_penalty"] for data in rewards_data_plot])
        plt.plot(x, [data["speed"] for data in rewards_data_plot])
        plt.plot(x, [data["robot_fell_penalty"] for data in rewards_data_plot])
        plt.plot(x, [data["robot_recovered_reward"]
                 for data in rewards_data_plot])
        plt.plot(x, [data["robot_on_ground_continuous_penalty"]
                     for data in rewards_data_plot])
        plt.plot(x, [data["relative_distance_from_target"]
                     for data in rewards_data_plot])
        plt.plot(x, [data["making_progress_towards_getting_up"]
                     for data in rewards_data_plot])
        plt.plot(x, [data["foot_contact_readings"]
                     for data in rewards_data_plot])
        plt.plot(x, [data["success_reward"]
                     for data in rewards_data_plot])
        plt.plot(x, [data["total_reward"]
                     for data in rewards_data_plot], ".", color="black", markersize=0.5)
        plt.legend([
            # "torque_force_penalty",
            "speed", "robot_fell_penalty", "robot_recovered_reward",
            "robot_on_ground_continuous_penalty", "relative_distance_from_target", "making_progress_towards_getting_up",
            "foot_contact_readings",
            "success_reward", "total_reward"])
        plt.savefig(
            f"{runtest_folder_name}/plots/rewards_epoch_{epoch}.png", dpi=300)
        plt.ylim(top=1, bottom=-1)
        plt.savefig(
            f"{runtest_folder_name}/plots/scaled_rewards_epoch_{epoch}.png", dpi=300)
        plt.clf()
        plt.close()

    rewards_data_plot = []

    final_epoch_coords, _ = p.getBasePositionAndOrientation(robot_id)

    avg_score = np.mean(score_history[-100:])
    score_history.append(epoch_acc_score)

    # log_divider()
    logger.info(f"EPOCH #{epoch}/{TOTAL_EPOCHS}:")
    logger.info(f"\t- Status: FINISHED")
    logger.info(f"\t- Started at: {epoch_start_timestamp}")
    logger.info(f"\t- Finished at: {get_date_time()}")
    logger.info(f"\t- Epoch score: {epoch_acc_score}")
    logger.info(f"\t- Average score: {avg_score}")
    logger.info(f"\t- Best score until now: {best_score}")
    logger.info(
        f"\t- Reached coords: [{final_epoch_coords[0]}, {final_epoch_coords[1]}]")
    log_divider()

    if avg_score > best_score:
        best_score = avg_score
        # log_divider()
        logger.info(
            f"EPOCH #{epoch}: Saving models due to better average performance")
        log_divider()
        agent.save_models()

    reset_env()

p.disconnect()
