import json
import os
from typing import List

import numpy as np

from pr2.solver import TaskSolverBase
from pr2.tcp import PersistentTcpClient, json2bin


import numpy as np
from typing import List, Dict
STATE_WALKING = 0
STATE_PICKING = 1
STATE_CIRCLING = 2

class DummyPlanner:

    def __init__(self, flag) -> None:

        self.cnt_ = 0
        self.state = 0
        self.state_start = 0;
    
        self.cmd_ = []

    def plan(self, obs: dict) -> List:

        obs_agent = obs["agent"]
        q_arm = obs_agent["joint_state"]["arms_positions"]
        attach = obs["pick"]

        vx, vy, theta = 0.0, 0.0, 0.0
        l_shoulder_y = q_arm[1]
        l_shoulder_z = 0
        l_elbow = 0
        pick = 0

        if (self.state == 0):
            # try move
            vx = 0.1
            vy = -0.002
            if (self.cnt_ - self.state_start > 7000):
                self.state = 1
                self.state_start = self.cnt_
                vx = 0.0

        elif (self.state == 1):
            # try touch
            if (attach):
                self.state = 2
                self.state_start = self.cnt_

            pick = 1
            l_shoulder_y = -0.5
            l_elbow = np.pi / 4
            vx = 0.0

        elif (self.state == 2):

            vx = 0.1
            if (self.cnt_ - self.state_start < 4000):
                vy = -0.005
                theta = -0.2
            
            
        cmd = [vx, vy, theta, -1, l_shoulder_y, l_shoulder_z, l_elbow, pick]
        self.cnt_ += 1
        self.cmd_.append(cmd)

        return cmd


class BipedWalkingCtrlClient(PersistentTcpClient):
    def send_request(self, msg):
        data_bin = json2bin(msg)
        return json.loads(self.send(data_bin).decode("ascii"))

    def get_cmd(self, obs, v_x, v_y, theta, state):
        obs_agent = obs["agent"]
        q_leg = obs_agent["joint_state"]["legs_positions"]
        dq_leg = obs_agent["joint_state"]["legs_velocities"]

        q_arm = obs_agent["joint_state"]["arms_positions"]
        dq_arm = obs_agent["joint_state"]["arms_velocities"]

        p_wb = obs_agent["body_state"]["world_pos"]
        quat_wb = obs_agent["body_state"]["world_orient"]
        v_wb = obs_agent["body_state"]["linear_velocities"]
        w_wb = obs_agent["body_state"]["angular_velocities"]

        msg = {
            "q_leg": q_leg.tolist(),
            "dq_leg": dq_leg.tolist(),
            "q_arm": q_arm.tolist(),
            "dq_arm": dq_arm.tolist(),
            "p_wb": p_wb.tolist(),
            "quat_wb": quat_wb.tolist(),
            "v_wb": v_wb.tolist(),
            "w_wb": w_wb.tolist(),
            "command": [v_x, v_y, theta],
            "change_state": state,
        }
        joint_efforts = self.send_request(msg)

        return joint_efforts


class TaskSolver(TaskSolverBase):
    def __init__(self) -> None:
        super().__init__()
        self.flag = True
        self.planner_ = DummyPlanner(flag=self.flag)
        self.ctrl_client_ = BipedWalkingCtrlClient(ip="0.0.0.0", port=8800)

    def next_action(self, obs: dict) -> dict:
        # plan for velocity cmd
        velocity_cmd = self.planner_.plan(obs)

        # call bipedal controller to get joint effort given a target velocity
        joint_efforts = self.ctrl_client_.get_cmd(
            obs, velocity_cmd[0], velocity_cmd[1], velocity_cmd[2], velocity_cmd[3]
        )

        # wrap joint effort into pr2 action format
        if self.flag:
            action = {
                "legs": {
                    "ctrl_mode": joint_efforts["mode"],
                    "joint_values": joint_efforts["effort"],
                    "stiffness": None,
                    "dampings": None,
                },
                "arms": {
                    "ctrl_mode": "position",
                    "joint_values": np.array(
                        [
                            np.pi / 4 + velocity_cmd[4],
                            0.0 + velocity_cmd[5],
                            0.0,
                            -np.pi * 2 / 3 + velocity_cmd[6],
                            np.pi / 4,
                            0.0,
                            0.0,
                            -np.pi * 2 / 3,
                        ]
                    ),
                },
                "pick": "left_hand" if velocity_cmd[7] == 1 else None,
            }
            if obs["pick"] is True:
                action["arms"]["stiffness"] = np.array([10] * 4 + [20] * 4)
                action["arms"]["dampings"] = np.array([1] * 4 + [2] * 4)
        else:
            action = {
                "legs": {
                    "ctrl_mode": "effort",
                    "joint_values": velocity_cmd[0:10],
                    "stiffness": None,
                    "dampings": None,
                },
                "arms": {
                    "ctrl_mode": "position",
                    "joint_values": np.array(
                        [
                            np.pi / 4 + velocity_cmd[10],
                            0.0 + velocity_cmd[11],
                            0.0,
                            -np.pi * 2 / 3 + velocity_cmd[12],
                            np.pi / 4,
                            0.0,
                            0.0,
                            -np.pi * 2 / 3,
                        ]
                    ),
                },
                "pick": "left_hand" if velocity_cmd[13] == 1 else None,
                "release": True if velocity_cmd[14] == 1 else None,
            }
            if velocity_cmd[13] == 1:
                action["arms"]["stiffness"] = np.array([10] * 8)
                action["arms"]["dampings"] = np.array([1] * 8)
                action["arms"]["joint_values"][1] += np.pi / 8

        return action
