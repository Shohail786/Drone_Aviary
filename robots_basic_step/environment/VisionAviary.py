import os
import numpy as np
from gym import spaces

from robots_basic_step.environment.BaseAviary import BaseAviary,DroneModel,Physics,ImageType

class VisionAviary(BaseAviary):
    """Multi-drone environment class for control applications using vision."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones:int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int =240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True
                 ):
        """Initialization of an aviary environment for control applications using vision.
        Attribute `vision_attributes` is automatically set to Trye when calling the
        superclass `__init__()` method.
        """

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=True
                         )
    def _actionSpace(self):
        """
        Return the action space of the environment

        :returns
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.
        """

        #Action vector-> P0, P1, P2, P3
        act_lower_bound=np.array([0.,0.,0.,0.])
        act_upper_bound=np.array([self.MAX_RPM,self.MAX_RPM,self.MAX_RPM,self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})

    def _observationSpace(self):
        """
        Returns the observation space of the environment

        :return:
        dict[str,dict[str,ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,),MultiBinary(NUM_DRONES),Box(H,W,4), Box(H,W), Box(H,W)}.

        """

        #Observation vector ##     X        Y      Z    Q1   Q2   Q3   Q4     R        P       Y        VX       VY      VZ       WR        WP       WY    P0  P1  P2  P3
        obs_lower_bound=np.array([-np.inf, -np.inf, 0., -1., -1., -1., -1., -np.pi, -np.pi, -np.pi,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound=np.array([np.inf,np.inf,np.inf,  1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,   np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i):spaces.Dict(
                                                {
                                                    "state": spaces.Box(low=obs_lower_bound,
                                                                   high=obs_upper_bound,
                                                                   dtype=np.float32
                                                                   ),

                                                    "neighbors": spaces.MultiBinary(self.NUM_DRONES),
                                                    "rgb": spaces.Box(low=0,
                                                                      high=255,
                                                                      shape=(self.IMG_RES[1],self.IMG_RES[0],4),
                                                                      dtype=np.uint8
                                                                      ),
                                                    "dep": spaces.Box(low=.01,
                                                                      high=1000,
                                                                      shape=(self.IMG_RES[1],self.IMG_RES[0]),
                                                                      dtype=np.float32
                                                                      ),
                                                    "seg": spaces.Box(low=0,
                                                                      high=100,
                                                                      shape=(self.IMG_RES[1],self.IMG_RES[0]),
                                                                      dtype=np.int
                                                                      )
                                                }) for i in range(self.NUM_DRONES)})

    def _computeObs(self):
        """
        returns the current observation of the environment.
        `getDroneStateVector()` 's implementation shows the value of the key "state",
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.


        :return:
        dict[str,dict[str,ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        adjacency_mat=self._getAdjacencyMatrix()
        obs={}
        for i in range(self.NUM_DRONES):
            if self.step_counter%self.IMG_CAPTURE_FREQ==0:
                self.rgb[i], self.dep[i], self.seg[i]=self._getDroneImages(i)

                #Printing observation to PNG example
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB, #ImageType.BW, ImageType.DEP, ImageType.SEG
                                     img_input=self.rgb[i],
                                      path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
            obs[str(i)]={ "state": self._getDroneStateVector(i),\
                          "neighbors": adjacency_mat[i,:],\
                          "rgb": self.rgb[i],\
                          "dep": self.dep[i],\
                          "seg": self.seg[i],\
                          }
        return obs

    def _preprocessAction(self,action):
        """
        Pre-process the action passed to `.step()` into motors' RPMs.

        Clips and coverts a dictionary into a 2D array.
        :param action: dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.
        :return:
        ndarray
            (NUM_DRONES,4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.
        """

        clipped_action=np.zeros((self.NUM_DRONES,4))
        for k,v in action.items():
            clipped_action[int(k),:]=np.clip(np.array(v),0,self.MAX_RPM)
        return clipped_action

    def _computeReward(self):
        """
        Computes the current reward value(s).
        Unused as this subclass is not meant for reinforcement learning

        :return:
            int
                Dummy value
        """
        return -1

    def _computeDone(self):
        """
        Computes the current done value(s).
        Unused as this subclass is not meant for reinforcement learning.
        :return:
        bool
            Dummy value
        """
        return False

    def _computeInfo(self):
        """
        Computes the current info dict(s).
        Unused as this subclass is not meant for reinforcement learning.
        :return:
        dict[str,int]
            Dummy value
        """

        return {"answer":42}  #calculated by the deep thought supercomputer in 7.5M years