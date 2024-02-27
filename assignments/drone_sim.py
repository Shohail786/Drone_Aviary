"""
Simulation script
The script uses the control defined in the drone_ctrl.py

Example
________
To run the simulation, type in the terminal
    $python drone_sim.py
"""

import time
import numpy as np
import pybullet as p
from robots_basic_step.environment.BaseAviary import DroneModel
from drone_ctrl import HW3Control
from robots_basic_step.environment.VisionAviary import VisionAviary
from robots_basic_step.utils.Logger import Logger
from robots_basic_step.utils.utils import sync

DURATION=10
"""int: The duration of the simulation in seconds"""
GUI=True
"""bool: whether to use the Pybullet graphical interface"""
RECORD=False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__=="__main__":
    #create environment
    ENV=VisionAviary(num_drones=1,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([[.0,.0,.15]]),
                     gui=GUI,
                     record=RECORD,
                     obstacles=False
                     )
    PYB_CLIENT = ENV.getPyBulletClient()  #return the PyBullet Client Id

    #Initialize the LOGGER
    LOGGER=Logger(logging_freq_hz=ENV.SIM_FREQ,num_drones=1)

    #Initialize the CONTROLLERS
    CTRL_0=HW3Control(env=ENV,
                      control_type=1,
                      )

    #Initialize the ACTION
    ACTION={}
    OBS=ENV.reset()
    STATE=OBS["0"]["state"]
    ACTION["0"]=CTRL_0.compute_control(current_position=STATE[0:3],
                                       current_velocity=STATE[10:13],
                                       current_rpy=STATE[7:10],
                                       target_position=np.append(STATE[0:3],STATE[9]),
                                       target_velocity=np.zeros(3),
                                       target_acceleration=np.zeros(3)
                                       )
    #Initialize the target trajectory
    TARGET_POSITION1=np.array([[4,.0,1.0,0] for i in range((DURATION*ENV.SIM_FREQ)//4)])
    TARGET_POSITION2=np.array([[4, 4,1.0,0] for i in range((DURATION*ENV.SIM_FREQ)//4)])
    TARGET_POSITION3=np.array([[0,4,1.0,0] for i in range((DURATION*ENV.SIM_FREQ)//4)])
    TARGET_POSITION4=np.array([[0,0,1.0,0] for i in range((DURATION*ENV.SIM_FREQ)//4)])
    TARGET_POSITION = np.concatenate((TARGET_POSITION1,TARGET_POSITION2,TARGET_POSITION3,TARGET_POSITION4))

    #print(TARGET_POSITION
    TARGET_VELOCITY=np.zeros([DURATION*ENV.SIM_FREQ,3])
    TARGET_ACCELERATION=np.zeros([DURATION*ENV.SIM_FREQ,3])
    q1=TARGET_POSITION+(np.array([.0,.0,.15,0]))
    q2=np.concatenate(([np.array([.0,.0,.15,0])],q1))

    #Derive the target trajectory to obtain the target velocities and accelerations
    TARGET_VELOCITY[1:,:]=(TARGET_POSITION[1:,0:3]-TARGET_POSITION[0:-1,0:3])/ENV.SIM_FREQ
    TARGET_ACCELERATION[1:,:]=(TARGET_VELOCITY[1:,:]-TARGET_VELOCITY[0:-1,:])/ENV.SIM_FREQ

    #Run the simulation
    START=time.time()

    p.addUserDebugParameter("speed",0,10)
    temp=[0,0,0]
    for i in range(0, DURATION*ENV.SIM_FREQ):

        #step the simulation
        OBS, _, _, _ =ENV.step(ACTION)

        # compute the control for drone 0
        STATE = OBS["0"]["state"]
        ACTION["0"] = CTRL_0.compute_control(  current_position=STATE[0:3],
                                               current_velocity=STATE[10:13],
                                               current_rpy=STATE[7:10],
                                               target_position=TARGET_POSITION[i,:]+np.array([.0,.0,.15,.0]),
                                               target_velocity=TARGET_VELOCITY[i,:],
                                               target_acceleration=TARGET_ACCELERATION[i,:]
                                               )

        physicsClientId=ENV.CLIENT
        #for i in range(0,DURATION*ENV.SIM_FREQ):
        p.addUserDebugLine(q2[i,0:3],q1[i,0:3],[1,0,0],1)
        p.addUserDebugText(str(q2[i,0:3]),q2[i,0:3],[1,0,0])
        p.addUserDebugLine(STATE[0:3],temp,[0,1,0],1)
        temp=STATE[0:3]

        #Log drone 0
        LOGGER.log(drone=0,timestamp=i/ENV.SIM_FREQ,state=STATE)

        #Printout
        if i%ENV.SIM_FREQ==0:
            ENV.render()

        #sync the simulation
        if GUI:
            sync(i,START,ENV.TIMESTEP)
    #Close the environment
    ENV.close()

    #Save the simulation results
    LOGGER.save()

    #Plot the simulation results
    LOGGER.plot()