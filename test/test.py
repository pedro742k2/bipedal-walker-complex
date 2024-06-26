from agent import Agent
from config import TARGET_POSITION, BATCH_SIZE, GAMMA, LR, MAX_ACTION_VALUE, MAX_RM_SIZE, MIN_ACTION_VALUE, TAU, TIMESTEPS_PER_EPOCH, TOTAL_EPOCHS
import numpy as np
import pybullet_data
import time
import pybullet as p
import argparse
import sys


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
    "--warmup", help="Warmup steps to run", type=int, default=0)
parser.add_argument(
    "--memory_size", help="Replay memory max size", type=int, default=MAX_RM_SIZE)
parser.add_argument(
    "--batch_size", help="Memory batch size for learning", type=int, default=BATCH_SIZE)
parser.add_argument(
    "--actor_lr", help="Actor neural network learning rate", type=float, default=LR)
parser.add_argument(
    "--critic_lr", help="Critic neural network learning rate", type=float, default=LR)
parser.add_argument(
    "--gamma", help="Gamma", type=float, default=GAMMA)
parser.add_argument(
    "--tau", help="Polyak averaging coefficient", type=float, default=TAU)
parser.add_argument(
    "--target_position", help="Target position (ex: [20, 20])", type=list, default=TARGET_POSITION)
parser.add_argument(
    "--gui", help="Use GUI mode", type=bool, default=True)

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

# or p.DIRECT for non-graphical version
physicsClientId = p.connect(p.GUI if gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.8)
# Spawn ground
plane_id = p.loadURDF("plane.urdf")
# Robot spawn position and orientation
cubeStartPos = [0, 0, 1.5]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# Spawn robot
robot_id = p.loadURDF("./robot.urdf",
                      cubeStartPos, cubeStartOrientation, useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
# Spawn target
tableId = p.loadURDF("table/table.urdf", basePosition=[
                     TARGET_POSITION[0], TARGET_POSITION[1], 0], physicsClientId=physicsClientId)

# Score history and best epoch score
score_history = []
best_score = 0

# Indicates if the robot, in the last timestep, was on the ground
was_robot_on_ground = False

# Joints info
num_joints = p.getNumJoints(robot_id)
joint_ids = [i for i in range(num_joints)]
joints_max_limit = []
joints_min_limit = []

OBSERVATION_SPACE_DIM = 100
ACTION_SPACE_DIM = num_joints
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
              state_space_dims=OBSERVATION_SPACE_DIM, actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, gamma=gamma, max_action=MAX_ACTION_VALUE, min_action=MIN_ACTION_VALUE, max_replay_size=memory_size, tau=tau, alpha=0.2, runtest_folder_name="./checkpoints")

agent.load_models()


def reset_env():
    # p.setJointMotorControlArray(robot_id, [i for i in range(
    #     num_joints)], p.POSITION_CONTROL, targetPositions=[0 for _ in range(num_joints)])
    p.resetBasePositionAndOrientation(
        robot_id, cubeStartPos, cubeStartOrientation)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, targetValue=0, targetVelocity=0)


def scale_value(x, lower_limit, upper_limit):
    return (abs(upper_limit)-abs(lower_limit)) * x + abs(lower_limit)


def rescale_actions(nn_actions):
    target_actions = []
    for current_action, lower_limit, upper_limit in list(zip(nn_actions, joints_min_limit, joints_max_limit)):
        # current_rescaled_action = current_action * \
        #     (abs(upper_limit) + abs(lower_limit)) - abs(lower_limit)

        target_actions.append(
            scale_value(current_action, lower_limit, upper_limit)
        )

    return target_actions


def is_state_success():
    (x, y, _), _ = p.getBasePositionAndOrientation(robot_id)
    X_TARGET, Y_TARGET = TARGET_POSITION
    return (abs(x - X_TARGET) < 0.05 and abs(y-Y_TARGET) < 0.05)


def get_relative_distance_from_target(current_x, current_y):
    X_TARGET, Y_TARGET = TARGET_POSITION

    return [abs(X_TARGET-current_x), abs(Y_TARGET-current_y)]


def get_contact_sensor_values():
    heel_1_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=3)), 1)
    foot_1_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=4)), 1)

    heel_2_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=8)), 1)
    foot_2_contact = min(len(p.getContactPoints(
        robot_id, plane_id, linkIndexA=9)), 1)

    return [
        max(foot_1_contact, heel_1_contact),
        max(foot_2_contact, heel_2_contact)
    ]


def is_state_truncated(timestep):
    return timestep == TIMESTEPS_PER_EPOCH-1


def get_distance_from_target_diff(state, new_state):
    past_state_distance_from_target = np.square(
        state[0] - TARGET_POSITION[0]) + np.square(state[1] - TARGET_POSITION[1])

    new_state_distance_from_target = np.square(
        new_state[0] - TARGET_POSITION[0]) + np.square(new_state[1] - TARGET_POSITION[1])

    distance_from_targets_diff = past_state_distance_from_target - \
        new_state_distance_from_target

    return distance_from_targets_diff


def get_current_observation():
    observation = np.array([])

    for joint_id in range(num_joints):
        joint_info = p.getLinkState(robot_id, joint_id)

        # Append joint position and orientation to observation (10 motors * 3 position dimensions = 30 items]
        joint_3d_position = joint_info[0]
        observation = np.append(observation, joint_3d_position)

        # Append joint position and orientation to observation (10 motors * 4 position dimensions = 40 items]
        joint_orientation = joint_info[1]
        observation = np.append(observation, joint_orientation)

        # Append joint rotation and velocity to observation (10 motors * 2 states = 20 items)
        joint_rotation, joint_velocity, _, _ = p.getJointState(
            robot_id, joint_id)
        observation = np.append(observation, [joint_rotation, joint_velocity])

    # Append robot linear and angular velocity to observation (2 velocities * 3 dimensions = 6 items)
    robot_linear_velocity, robot_angular_velocity = p.getBaseVelocity(
        robot_id)
    observation = np.append(
        observation, [robot_linear_velocity, robot_angular_velocity])

    # Append relative target position (|x-target|, |y-target|) (2 items)
    (current_x, current_y, _), _ = p.getBasePositionAndOrientation(robot_id)
    observation = np.append(
        observation,
        get_relative_distance_from_target(
            current_x, current_y)
    )

    # Append contact sensors (binary array with two elements: feet touching the ground) (2 items)
    observation = np.append(observation, get_contact_sensor_values())

    # Total of 100 elements
    return observation


def get_last_timestep_t():
    return last_timestep_t


def is_robot_on_ground():
    return len(p.getContactPoints(robot_id, plane_id, linkIndexA=-1)) > 0


def update_last_timestep_t():
    global last_timestep_t
    last_timestep_t = time.time()


def get_was_robot_on_ground():
    return was_robot_on_ground


def update_was_robot_on_ground(flag: bool):
    global was_robot_on_ground
    was_robot_on_ground = flag


def robot_fell(prev_fell_flag, current_fell_flag):
    """If the robot wasn't on the ground, but now is"""
    return not prev_fell_flag and current_fell_flag


def recovered_from_fall(prev_fell_flag, current_fell_flag):
    """If the robot recovered from a fall"""
    return prev_fell_flag and not current_fell_flag


def get_reward(past_position, new_position, timestep, scale_factor=1):
    timestep_reward = 0

    prev_fell_flag = get_was_robot_on_ground()
    current_fell_flag = is_robot_on_ground()

    # Get travelled distance
    distance_traveled = np.linalg.norm(
        np.array(past_position[:2]) - np.array(new_position[:2])
    )

    timestep_reward += distance_traveled

    # Calculate robot speed
    last_timestep_t = get_last_timestep_t()
    speed = (distance_traveled / (time.time() - last_timestep_t)) * 1e-1
    timestep_reward += speed

    if robot_fell(prev_fell_flag, current_fell_flag):
        timestep_reward -= 0.5
    elif recovered_from_fall(prev_fell_flag, current_fell_flag):
        timestep_reward += 0.5

    if current_fell_flag:
        timestep_reward -= 0.05

    if is_state_success():
        timestep_reward += 300

    # TODO: TEST -> RUN WITH MIN BEING 0 AND WITHOUT RESTRICTION
    timestep_reward += max(
        0, get_distance_from_target_diff(past_position, new_position)
    )
    # timestep_reward += get_distance_from_target_diff(
    #     past_position, new_position)

    update_last_timestep_t()
    update_was_robot_on_ground(current_fell_flag)

    return timestep_reward * scale_factor


# Configure initial timestamp
last_timestep_t = time.time()


for epoch in range(TOTAL_EPOCHS):
    epoch_acc_score = 0

    for timestep in range(TIMESTEPS_PER_EPOCH):
        position, _ = p.getBasePositionAndOrientation(robot_id)
        # p.resetDebugVisualizerCamera(3, 45, -25, position)

        # Get current observation
        current_observation = get_current_observation()

        # Get actions | range: [-1, 1]
        actions = agent.select_actions(current_observation, False)
        rescaled_actions = rescale_actions(actions)

        # Apply actions
        for i in joint_ids:
            p.setJointMotorControl2(robot_id, i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=rescaled_actions[i], maxVelocity=100, force=300)

        # Perform actions in the environment
        p.stepSimulation()

        if gui:
            time.sleep(1./240.)

        # Get new observation
        new_observation = get_current_observation()

        # Get new robot positions
        new_position, _ = p.getBasePositionAndOrientation(robot_id)

        # Get timestep reward
        reward = get_reward(position, new_position, timestep)

        # Add reward to epoch accumulated score
        epoch_acc_score += reward

        # Check if epoch truncated
        terminal_flag = is_state_truncated(i)

        # agent.remember(current_observation, actions, reward,
        #                new_observation, terminal_flag)

        # agent.learn()

        if is_state_success():
            break

    avg_score = np.mean(score_history[-100:])
    score_history.append(epoch_acc_score)

    if avg_score > best_score:
        best_score = avg_score

    reset_env()

p.disconnect()
