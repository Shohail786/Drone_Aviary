"""
The controller used the simulation in the file `drone_sim.py`
Example
_______
To run the simulation, type in the terminal: $ python drone_sim.py

Notes
_____
ToDOs
    Search for word "Objective" this file (there are 4 occurrences)
    Fill appropriate values in the 3 by 3 matrix self.matrix_u2m.
    Compute u_1 for the linear controller and the second nonlinear one Compute u_2
"""

import numpy as np
from robots_basic_step.environment.BaseAviary import BaseAviary

class HW3Control():
    """
    control class
    """
    def __init__(self,
                 env:BaseAviary,
                 control_type:int=0
                 ):
        """
        Initialization of class HW3Control
        :param env: BaseAviary
            The PyBullet-based simulation environment
        :param control_type: int, optiona;
            Choose between implementation of the u1 computation
        """
        self.g=env.G
        """float: Gravity acceleration, in meters per second squared"""
        self.mass=env.M
        """float:The mass of quad from environment"""
        self.inertia_xx=env.J[0][0]
        """float:The inertia of quad around x axis"""
        self.inertia_yy=env.J[1][1]
        """float:The inertia of quad around y axis"""
        self.inertia_zz=env.J[2][2]
        """float: The inertia of quad around z axis"""
        self.arm_length=env.L
        """float: The length of the arm of the drone"""
        self.timestep=env.TIMESTEP
        """float: Simulation and control timestep"""
        self.last_rpy=np.zeros(3)
        """ndarray: Store the last roll, pitch and yaw"""
        self.kf_coeff=env.KF
        """float: RPMs to force coefficient"""
        self.km_coeff=env.KM
        """float: RPMs to torque coefficient"""
        self.CTRL_TYPE=control_type
        """int: Flag switching between implementation of u1"""
        self.p_coeff_position={}
        """dict[str,float]: Proportional coefficient(s) for position control"""
        self.d_coeff_position={}
        """dict[str,float]: Derivative coefficient(s) for position control"""

        #TODO parts
        #Objectives
        self.matrix_u2rpm=np.array([ [2, 1, 1],
                                     [0, 1, -1],
                                     [2, -1, -1]
                                    ])
        """ndarray: (3,3)-shaped array of ints to determine motor rpm from force and torque"""

        self.matrix1_u2rpm=np.array([[0.3,0,0.25],
                                     [0.3,0.5,-0.25],
                                     [0.3,-0.5,-0.25]
                                     ])
        self.matrix_u2rpm_inv=np.linalg.inv(self.matrix_u2rpm)

        self.p_coeff_position["z"]=0.7*0.7
        self.d_coeff_position["z"]=2*0.5*0.7
        #
        self.p_coeff_position["y"]=0.7*0.7
        self.d_coeff_position["y"]=2*0.5*0.7
        #
        self.p_coeff_position["x"]=0.7*0.7
        self.d_coeff_position["x"]=2*0.5*0.7
        #
        self.p_coeff_position["r"]=2*0.7*0.7
        self.d_coeff_position["r"]=2*2*2.5*0.7
        #
        self.p_coeff_position["p"]=2*0.7*0.7
        self.d_coeff_position["p"]=2*2*2.5*0.7
        #
        self.p_coeff_position["yaw"]=0.7*0.7
        self.d_coeff_position["yaw"]=2*2.5*0.7

        self.reset()

    def reset(self):
        """
        Resets the controller counter
        :return:
        """
        self.control_counter=0

    def compute_control(self,
                        current_position,
                        current_velocity,
                        current_rpy,
                        target_position,
                        target_velocity=np.zeros(3),
                        target_acceleration=np.zeros(3)
                        ):
        """
        compute the target state through calculating the propellers' RPM, given the current state
        :param current_position: ndarray
            (3,)-shaped array of floats containing global x,y,z, in meters.
        :param current_velocity: ndarray
            (3,)-shaped array of floats containing global vx,vy,vz in m/s.
        :param current_rpy: ndarray
            (3,)-shaped array of floats containing roll,pitch,yaw in rad.
        :param target_position: ndarray
            (3,)-shaped array of floats containing global x,y,z in meters.
        :param target_velocity: ndarray, optional
            (3,)-shaped array of floats containing the global, in m/s.
        :param target_acceleration: ndarray, optional
            (3,)-shaped array of floats containing global, in m/s^2
        :return:
            (4,)-shaped array of ints containing the desired RPMs of each propeller.
        """
        self.control_counter+=1

        #compute the roll,pitch and yaw rates
        current_rpy_dot=(current_rpy-self.last_rpy)/self.timestep

        #compute the PD control in y,z
        x_ddot=self.pd_control(target_position[0],
                               current_position[0],
                               target_velocity[0],
                               current_velocity[0],
                               target_acceleration[0],
                               "x"
                               )
        y_ddot=self.pd_control(target_position[1],
                               current_position[1],
                               target_velocity[1],
                               current_velocity[1],
                               target_acceleration[1],
                               "y"
                               )
        z_ddot=self.pd_control(target_position[2],
                               current_position[2],
                               target_velocity[2],
                               current_velocity[2],
                               target_acceleration[2],
                               "z"
                               )

        #calculate the desired roll and rates given by PD
        desired_roll=np.arcsin(-y_ddot/((self.g+z_ddot)/(np.cos(current_rpy[1])*np.cos(current_rpy[0]))))

        #print("desired_roll",desired_roll)
        desired_roll_dot=(desired_roll-current_rpy[0])/0.004
        self.old_roll=desired_roll
        self.old_roll_dot=desired_roll_dot
        roll_ddot=self.pd_control(desired_roll,
                                  current_rpy[0],
                                  desired_roll_dot,
                                  current_rpy_dot[0],
                                  0,
                                  "r"
                                  )
        #calculate the desired pitch and rates given by PD
        desired_pitch=np.arcsin(x_ddot/((self.g+z_ddot)/(np.cos(current_rpy[1])*np.cos(current_rpy[0]))))
        #print("desired_pitch",desired_pitch)
        desired_pitch_dot=(desired_pitch-current_rpy[1])/0.004
        self.old_pitch=desired_pitch
        self.old_pitch_dot=desired_pitch_dot
        pitch_ddot=self.pd_control(desired_pitch,
                                   current_rpy[1],
                                   desired_pitch_dot,
                                   current_rpy_dot[1],
                                   0,
                                   "p"
                                   )
        #calculate desired yaw and rates given by PD
        desired_yaw=target_position[3]
        #print("desired_yaw_dot",desired_yaw_dot)
        desired_yaw_dot = (desired_yaw - current_rpy[2]) / 0.004
        self.old_yaw=desired_yaw
        self.old_yaw_dot=desired_yaw_dot
        yaw_ddot=self.pd_control(desired_yaw,
                                 current_rpy[2],
                                 desired_yaw_dot,
                                 current_rpy_dot[2],
                                 0,
                                 "yaw"
                                 )
        #print("yaw_ddot",yaw_ddot)

        v=np.array([[-z_ddot],[roll_ddot],[pitch_ddot],[yaw_ddot]])
        b=np.array([[self.g],
                    [current_rpy_dot[1]*current_rpy_dot[2]*((self.inertia_yy-self.inertia_zz)/self.inertia_xx)],
                    [current_rpy_dot[0]*current_rpy_dot[2]*((self.inertia_zz-self.inertia_xx)/self.inertia_yy)],
                    [current_rpy_dot[0]*current_rpy_dot[1]*((self.inertia_xx-self.inertia_yy)/self.inertia_zz)]
                    ])
        D=np.array([[np.cos(current_rpy[0])*np.cos(current_rpy[1])*(-1/self.mass),0,0,0],
                    [0,1/self.inertia_xx,0,0],
                    [0,0,1/self.inertia_yy,0],
                    [0,0,0,1/self.inertia_zz]
                    ])
        beta=np.linalg.inv(D)
        #print(beta)
        alpha=-np.dot(beta,b)
        u1=alpha+np.dot(beta,v)
        f_t=u1[0][0]/self.kf_coeff
        t_x=u1[1][0]/(self.arm_length*self.kf_coeff)
        t_y=u1[2][0]/(self.arm_length*self.kf_coeff)
        t_z=u1[3][0]/(self.arm_length*self.kf_coeff)
        print(f_t,t_x,t_y,t_z)

        #calculate thrust and moment given the PD input
        if self.CTRL_TYPE==0:
            u_1=self.mass*(self.g+self.z_ddot)

        elif self.CTRL_TYPE==1:
            u_1=self.mass*(((self.g+z_ddot)/(np.cos(current_rpy[1])*np.cos(current_rpy[0]))))

        elif self.CTRL_TYPE==2:
            u_1=self.mass*np.sqrt(y_ddot**2+x_ddot**2+(self.g+z_ddot)**2)

        u_21=self.inertia_xx*roll_ddot
        u_22=self.inertia_yy*pitch_ddot
        u_23=self.inertia_zz*yaw_ddot

        x=(u_21)/(self.arm_length*self.kf_coeff)
        y=(u_22)/(self.arm_length*self.kf_coeff)
        z=(u_23)/(self.arm_length*self.kf_coeff)
        f=(u_1)/self.kf_coeff
        print("x",x)
        print("y",y)
        print("z",z)
        print("f",f)
        f_tm=(f+f_t)/2
        if(f<f_tm):
            F=f_t
        else:
            F=f
        print("F",F)
        propellers_0_rpm=np.sqrt(np.abs((f_t-2*t_y-t_z)/4))
        propellers_1_rpm=np.sqrt(np.abs((f_t+2*t_x+t_z)/4))
        propellers_2_rpm=np.sqrt(np.abs((f_t+2*t_y-t_z)/4))
        propellers_3_rpm=np.sqrt(np.abs((f_t-2*t_x+t_z)/4))

        print("propellers_0_rpm",propellers_0_rpm)
        print("propellers_1_rpm",propellers_1_rpm)
        print("propellers_2_rpm",propellers_2_rpm)
        print("propellers_3_rpm",propellers_3_rpm)

        #print the relevant output
        if self.control_counter%(1/self.timestep)==0:
            print("current_position",current_position)
            print("current_velocity",current_velocity)
            print("target_position",target_position)
            print("target_velocity",target_velocity)
            print("target_acceleration",target_acceleration)

        #store the last step's roll, pitch, and yaw
        self.last_rpy=current_rpy

        return np.array([propellers_0_rpm, propellers_1_rpm,
                         propellers_2_rpm,propellers_3_rpm])



    def pd_control(self,
                   desired_position,
                   current_position,
                   desired_velocity,
                   current_velocity,
                   desired_acceleration,
                   opt
                   ):
        """
        Compute PD control for the acceleration minimizing position error
        :param desired_position: float
            Desired global position
        :param current_position: float
            current global position
        :param desired_velocity: float
            desired global velocity
        :param current_velocity: float
            current global velocity
        :param desired_acceleration: float
            desired global acceleration
        :param opt:
        :return: float
            The commanded acceleration
        """
        u=desired_acceleration+ \
            self.d_coeff_position[opt]*(desired_velocity-current_velocity)+\
            self.p_coeff_position[opt]*(desired_position-current_position)

        return u
