# Franka "open the top drawer, then pick up a cube and place it inside" task,
# implemented as an Isaac Lab DirectRLEnv.
#
# The scene setup, robot/cabinet asset configs, grasp-frame math and the
# open-drawer reward are taken faithfully from Isaac Lab's proven
# `Isaac-Franka-Cabinet-Direct-v0` task (so it loads and the open phase trains).
# Everything for the cube pick-and-place stage is added on top.
#
# Long-horizon caveat: open + pick + place is hard for pure RL. The reward is
# staged (open first, then place) and the README explains the curriculum recipe.

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


@configclass
class FrankaCabinetPlaceEnvCfg(DirectRLEnvCfg):
    # --- env ---
    episode_length_s = 10.0  # longer than open-only (need time to also place)
    decimation = 2
    action_space = 9        # 7 arm joints + 2 finger joints
    # obs: dof_pos_scaled(9) + dof_vel(9) + grasp->handle(3) + drawer_pos(1)
    #      + drawer_vel(1) + cube_pos(3) + cube->grasp(3) + cube->place(3) = 32
    observation_space = 32
    state_space = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # --- robot (inline cfg, facing the cabinet at x=1) ---
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=80.0, damping=4.0
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=80.0, damping=4.0
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
            ),
        },
    )

    # --- cabinet ---
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            # stiffness=0 so the drawer has no return-spring: once pulled open it
            # stays open (needed so the arm can release it and go grab the cube).
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0, stiffness=0.0, damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0, stiffness=10.0, damping=2.5,
            ),
        },
    )

    # --- small side pedestal (kinematic: fixed in place, collidable) ---
    # Small footprint and pushed out to +y so its inner edge (y=0.38) is clear of
    # the arm's working area around y=0 -- otherwise a link jams against it.
    table_height = 0.30
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, table_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.48, table_height / 2)),
    )

    # --- cube to be placed into the drawer (starts ON the pedestal) ---
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.045, 0.045, 0.045),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.9)),
        ),
        # just above the pedestal top (0.30) so it settles onto the surface
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.48, 0.33)),
    )

    # --- ground ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    # --- control / scales ---
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # --- reward scales (open stage: matches the stock cabinet task) ---
    dist_reward_scale = 1.5      # gripper -> drawer handle
    rot_reward_scale = 1.5       # gripper/drawer axis alignment
    around_handle_reward_scale = 0.25  # bonus for fingers straddling the handle
    open_reward_scale = 10.0     # drawer openness (coupled with being around handle)
    finger_reward_scale = 2.0    # fingers close to + straddling the handle (key for pulling)
    action_penalty_scale = 0.05

    # --- place stage ---
    # Set enable_place=False to learn ONLY to open the drawer first (recommended:
    # train open, then load that checkpoint and flip this on). When False, the
    # episode also terminates on a fully-open drawer, exactly like the stock task.
    enable_place = False
    open_threshold = 0.20        # m, drawer "open enough" to switch to placing
    reach_cube_scale = 2.0       # gripper -> cube
    lift_scale = 10.0            # reward lifting the cube off the table once near it
    place_scale = 4.0            # cube -> place target (inside drawer)
    place_success_dist = 0.06    # m, cube counts as "inside"
    place_bonus = 5.0
    open_achieved_bonus = 2.0    # constant value of having the drawer open


class FrankaCabinetPlaceEnv(DirectRLEnv):
    cfg: FrankaCabinetPlaceEnvCfg

    def __init__(self, cfg: FrankaCabinetPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos, xformable, device):
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            px, py, pz = world_pos[0] - env_pos[0], world_pos[1] - env_pos[1], world_pos[2] - env_pos[2]
            qx, qy, qz = world_quat.imaginary[0], world_quat.imaginary[1], world_quat.imaginary[2]
            qw = world_quat.real
            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # --- grasp frames (hand and drawer handle), taken from the stock task ---
        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        grasp_rot, grasp_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        grasp_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = grasp_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = grasp_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]
        self.drawer_joint_idx = self._cabinet.find_joints("drawer_top_joint")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_target_pos = torch.zeros((self.num_envs, 3), device=self.device)  # inside the drawer

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self._cube = RigidObject(self.cfg.cube)
        self._table = RigidObject(self.cfg.table)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.rigid_objects["table"] = self._table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ---------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # No success-termination in either phase: the per-step reward is positive,
        # so ending early would forfeit reward and discourage finishing the task.
        return torch.zeros_like(truncated), truncated

    # -------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        drawer_pos = self._cabinet.data.joint_pos[:, self.drawer_joint_idx]
        lfinger = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        rfinger = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        # distance gripper-grasp -> handle-grasp
        d = torch.norm(self.robot_grasp_pos - self.drawer_grasp_pos, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward = dist_reward ** 2
        dist_reward = torch.where(d <= 0.02, 2.0 * dist_reward, dist_reward)

        # alignment of gripper axes with the drawer axes
        axis1 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        axis2 = tf_vector(self.drawer_grasp_rot, self.drawer_inward_axis)
        axis3 = tf_vector(self.robot_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.drawer_grasp_rot, self.drawer_up_axis)
        dot1 = (axis1 * axis2).sum(dim=-1)
        dot2 = (axis3 * axis4).sum(dim=-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # fingers must straddle the handle (left above, right below) AND be close.
        # This is the term whose absence made the first version just hover.
        handle_z = self.drawer_grasp_pos[:, 2]
        straddle = (lfinger[:, 2] > handle_z) & (rfinger[:, 2] < handle_z)
        around_handle_reward = torch.where(straddle, torch.ones_like(d) * 0.5, torch.zeros_like(d))
        lfinger_dist = torch.abs(lfinger[:, 2] - handle_z)
        rfinger_dist = torch.abs(rfinger[:, 2] - handle_z)
        finger_dist_reward = torch.where(
            straddle, (0.04 - lfinger_dist) + (0.04 - rfinger_dist), torch.zeros_like(d)
        )

        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)

        # gripper -> handle shaping (this is what anchors the arm at the handle;
        # it must be switched OFF once the drawer is open, or the arm never leaves)
        gripper_to_handle = (
            self.cfg.dist_reward_scale * dist_reward
            + self.cfg.rot_reward_scale * rot_reward
            + self.cfg.around_handle_reward_scale * around_handle_reward
            + self.cfg.finger_reward_scale * finger_dist_reward
        )
        # openness is rewarded in BOTH phases (coupled with being around the handle)
        openness = self.cfg.open_reward_scale * (drawer_pos * around_handle_reward + drawer_pos)

        # ---- open-only training (phase 1) ----
        if not self.cfg.enable_place:
            r = gripper_to_handle + openness - action_penalty
            r = torch.where(drawer_pos > 0.01, r + 0.5, r)
            r = torch.where(drawer_pos > 0.20, r + around_handle_reward, r)
            r = torch.where(drawer_pos > 0.39, r + 2.0 * around_handle_reward, r)
            return r

        # ---- place shaping (phase 2: only meaningful once the drawer is open) ----
        is_open = drawer_pos > self.cfg.open_threshold
        d_cube = torch.norm(self.robot_grasp_pos - self.cube_pos_w, dim=-1)
        reach_cube = 1.0 - torch.tanh(d_cube / 0.10)
        near_cube = (d_cube < 0.06).float()
        # height of the cube above the table top -> rewards actually lifting it
        cube_height = torch.clamp(self.cube_pos_w[:, 2] - self.cfg.table_height, min=0.0)
        lift = self.cfg.lift_scale * torch.clamp(cube_height, max=0.25) * near_cube
        d_place = torch.norm(self.cube_pos_w - self.place_target_pos, dim=-1)
        place_reward = 1.0 - torch.tanh(d_place / 0.15)
        placed_bonus = (d_place < self.cfg.place_success_dist).float()

        place_stage = (
            self.cfg.reach_cube_scale * reach_cube
            + lift
            + self.cfg.place_scale * place_reward
            + self.cfg.place_bonus * placed_bonus
        )

        # Before open: reward opening (handle shaping). After open: DROP the handle
        # shaping so the arm is free to leave the handle, and reward placing instead.
        reward = openness - action_penalty + torch.where(
            is_open,
            place_stage + self.cfg.open_achieved_bonus,
            gripper_to_handle,
        )
        return reward

    # ---------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet (closed)
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # cube: drop near the start pose with small xy noise
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0:3] += self.scene.env_origins[env_ids]
        cube_state[:, 0:2] += sample_uniform(-0.05, 0.05, (len(env_ids), 2), self.device)
        self._cube.write_root_pose_to_sim(cube_state[:, 0:7], env_ids=env_ids)
        self._cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]), env_ids=env_ids)

        self._compute_intermediate_values(env_ids)

    # ----------------------------------------------------------- observations
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        )
        to_handle = self.drawer_grasp_pos - self.robot_grasp_pos
        cube_to_grasp = self.cube_pos_w - self.robot_grasp_pos
        cube_to_place = self.place_target_pos - self.cube_pos_w
        cube_pos_b = self.cube_pos_w - self.scene.env_origins

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_handle,
                self._cabinet.data.joint_pos[:, self.drawer_joint_idx].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, self.drawer_joint_idx].unsqueeze(-1),
                cube_pos_b,
                cube_to_grasp,
                cube_to_place,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------- auxiliary values
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]

        robot_grasp_rot, robot_grasp_pos = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
        )
        drawer_grasp_rot, drawer_grasp_pos = tf_combine(
            drawer_rot, drawer_pos, self.drawer_local_grasp_rot[env_ids], self.drawer_local_grasp_pos[env_ids]
        )
        self.robot_grasp_rot[env_ids] = robot_grasp_rot
        self.robot_grasp_pos[env_ids] = robot_grasp_pos
        self.drawer_grasp_rot[env_ids] = drawer_grasp_rot
        self.drawer_grasp_pos[env_ids] = drawer_grasp_pos

        # place target = a point inside the drawer: from the handle, move inward
        # (toward the drawer body) and up a little so the cube clears the floor.
        inward_w = tf_vector(drawer_grasp_rot, self.drawer_inward_axis[env_ids])
        up_w = tf_vector(drawer_grasp_rot, self.drawer_up_axis[env_ids])
        self.place_target_pos[env_ids] = drawer_grasp_pos + 0.15 * inward_w + 0.04 * up_w

        # cube position (world)
        if not hasattr(self, "cube_pos_w"):
            self.cube_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_pos_w[env_ids] = self._cube.data.root_pos_w[env_ids]
