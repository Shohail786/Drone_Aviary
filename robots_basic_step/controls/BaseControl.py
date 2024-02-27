import numpy as np
from robots_basic_step.environment.BaseAviary import BaseAviary

class BaseControl(object):
    """
    Base class for control.
    Implements `__init__()`, `reset()`, and `interface computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.
    """
    def __init__(self,
                 env:BaseAviary
                 ):
        """
        Common control classes __init__ method.
        :param
            env: BaseAviary-> The simulation environment to control.
        """
        # set the general use constants
        self.DRONE_MODEL=env.DRONE_MODEL
        """int: The number of drones in the simulation environment."""
        self.GRAVITY=env.GRAVITY
        """float: The gravitational force (M*g) acting on the drone."""
        self.KF=env.KF
        """float: The coefficient converting RPMs into thrust."""
        self.KM=env.KM
        """float: The coefficient converting RPMs into torque."""
        self.reset()

    def reset(self):
        """Reset the control classes.
        A general use counter is set to zero.

        """
        self.control_counter=0

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_ang_vel=np.zeros(3)
                                ):
        """
        Interface method using `computeControl`.
        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step(),

        :param
            control_timestep: float
                The time step at which control is computed.
        :param
            state: ndarray
                (20,)-shaped array of floats containing the current state of the drone.
        :param
            target_pos: ndarray
                (3,1)-shaped array of floats containing the desired position.
        :param
            target_rpy: ndarray, optional
                (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        :param
            target_vel: ndarray, optional
                (3,1)-shaped array of floats containing the desired velocity.
        :param
            target_ang_vel: ndarray, optional
                (3,1)-shaped array of floats containing the desired angular velocity
        :return: parameters with value to compute control
        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_vel=target_vel,
                                   target_ang_vel=target_ang_vel
                                   )

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_ang_vel=np.zeros(3)
                       ):
        """
        Abstract method to compute the control action for a single drone.
        It must be implemented by each subclass of `BaseControl`.

        :param control_timestep: float
            The time step at which control is computed.
        :param cur_pos: ndarray
            (3,1)-shaped array of floats containing the current position.
        :param cur_quat: ndarray
            (4,1)-shaped array of float containing the current orientation as a quaternion.
        :param cur_vel: ndarray
            (3,1)-shaped array of floats containing the current velocity
        :param cur_ang_vel: ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        :param target_pos: ndarray
            (3,1)-shaped array of floats containing the desired position.
        :param target_rpy: ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        :param target_vel: ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        :param target_ang_vel: ndarray, optional
            (3,1)-shaped array of floats containing the desired angular velocity.
        :return:
        """

        raise NotImplementedError