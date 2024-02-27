import os
from datetime import datetime
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """
    A class for logging and visualization

    Stores, saves to file and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """
    def __init__(self,
                 logging_freq_hz:int,
                 num_drones:int=1,
                 duration_sec:int=0
                 ):
        """
        Logger class __init__ method

        :param logging_freq_hz: int
            Logging frequency in Hz
        :param num_drones: int, optional
            Number of drones
        :param duration_sec: int, optional
            Used to preallocate the log arrays(improves performance)
        :return:
        """
        self.LOGGING_FREQ_HZ=logging_freq_hz
        self.NUM_DRONES=num_drones
        self.PREALLOCATED_ARRAYS=False if duration_sec==0 else True
        self.counters=np.zeros(num_drones)
        self.timestamps=np.zeros((num_drones,duration_sec*self.LOGGING_FREQ_HZ))
        self.states=np.zeros((num_drones,16,duration_sec*self.LOGGING_FREQ_HZ))

        ##16 states: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
        ##           ang_vel_x, ang_vel_y, ang_vel_z, rpm0, rpm1, rpm2, rpm3

        self.controls=np.zeros((num_drones,12,duration_sec*self.LOGGING_FREQ_HZ))

        ##12 control targets: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
        ##                    ang_vel_x, ang_vel_y, ang_vel_z

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12)
            ):
        """
        Logs entries for a single simulation step, of a single drone.
        :param drone: int
                Id of the drone associated to the log entry
        :param timestamp: float
                Timestamp of the log in simulation clock.
        :param state: ndarray
                (20,)-shaped array of floats containing the drone's state
        :param control: ndarray, optiona;
                (12,)-shaped array of floats containing the drone's control target
        :return:

        """

        if drone<0 or drone>=self.NUM_DRONES or timestamp<0 or len(state)!=20 or len(control)!=12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter=int(self.counters[drone])

        #Add rows to the matrices of a counter exceeds their size
        if current_counter>=self.timestamps.shape[1]:
            self.timestamps=np.concatenate((self.timestamps,np.zeros((self.NUM_DRONES,1))),axis=1)
            self.states=np.concatenate((self.states,np.zeros((self.NUM_DRONES,16,1))),axis=2)
            self.controls=np.concatenate((self.controls,np.zeros((self.NUM_DRONES,12,1))),axis=2)

        #Advance a counter is the matrices have overgrown it
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1]>current_counter:
            current_counter=self.timestamps.shape[1]-1

        #Log the information and increase the counter
        self.timestamps[drone,current_counter]=timestamp
        #np.hstack creates the array in the sequence of states that is provided
        self.states[drone,:,current_counter]=np.hstack([state[0:3],state[10:13],state[7:10],state[13:20]])
        self.controls[drone,:,current_counter]=control
        self.controls[drone]=current_counter+1

    def save(self):
        """
        Save the logs to file
        :param self:
        :return:
        """
        with open(os.path.dirname(os.path.abspath(__file__))+"/../../files/logs/save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy","wb") as out_file:
            np.save(out_file,self.timestamps)
            np.save(out_file,self.states)
            np.save(out_file,self.controls)





