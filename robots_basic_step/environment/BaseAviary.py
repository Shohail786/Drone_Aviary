import os
import time
import collections
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as etxml
from PIL import Image
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gym

print("hello")
class DroneModel(Enum):

    """Drone Models enumeration class."""
    CF2X="cf2x"
    CF2P="cf2p"
    HB="hb"
    Trixy="trixy"
    Qrotor="quadrotor"
    Racecar="racecar"

class Physics(Enum):
    """Physics implementations enumeration class."""
    PYB="pyb"
    DYN="dyn"
    PYB_GND="pyb_gnd"
    PYB_DRAG="pyb_drag"
    PYB_DW="pyb_dw"
    PYB_GND_DRAG_DW="pyb_gnd_drag_dw"

class ImageType(Enum):
    """Camera capture image type enumeration class."""
    RGB=0       #Red, green, blue (and alpha)
    DEP=1       #Depth
    SEG=2       #Segmentation by object id
    BW=3        #Black and White

class BaseAviary(gym.Env):
    """Base class for 'drone aviary' Gym environments. """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius:float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps:int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=True,
                 dynamics_attributes=False
                 ):
        #constants
        self.G=9.8
        self.RAD2DEG=180/np.pi
        self.DEG2RAD=np.pi/180
        self.SIM_FREQ=freq
        self.TIMESTEP=1./self.SIM_FREQ
        self.AGGR_PHY_STEPS=aggregate_phy_steps
        #parameters
        self.NUM_DRONES=num_drones
        self.NEIGHBOURHOOD_RADIUS=neighbourhood_radius
        #options
        self.DRONE_MODEL=drone_model
        self.GUI=gui
        self.RECORD=record
        self.PHYSICS=physics
        self.OBSTACLES=obstacles
        self.USER_DEBUG=user_debug_gui
        self.URDF=self.DRONE_MODEL.value+".urdf"
        print("URDF>>", self.URDF)
        #load the drone properties from the .urdf file
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_inv, \
        self.KF, \
        self.KM, \
        self.COLLISION_H, \
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3=self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n"
              "[INFO] m {:f}, L {:f},\n [INFO] ixx {:f}, iyy {:f}, izz {:f},\n [INFO] kf {:f}, km {:f}, \n"
              "[INFO] t2w {:f}, max_speed_kmh {:f},\n [INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n"
              "[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \n [INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH,
            self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #compute constants
        self.GRAVITY=self.G*self.M
        self.HOVER_RPM=np.sqrt(self.GRAVITY/(4*self.KF))
        self.MAX_RPM=np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY)/(4*self.KF))
        self.MAX_THRUST=(4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE=(self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE=(2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP=0.25*self.PROP_RADIUS* np.sqrt((15*self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF)/ self.MAX_THRUST)
        #create attributes for vision tasks
        self.VISION_ATTR=vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES=np.array([64,48])
            self.IMG_FRAME_PER_SEC=24
            self.IMG_CAPTURE_FREQ=int(self.SIM_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb=np.zeros(((self.NUM_DRONES,self.IMG_RES[1],self.IMG_RES[0],4)))
            self.dep=np.ones(((self.NUM_DRONES,self.IMG_RES[1],self.IMG_RES[0])))
            self.seg=np.zeros(((self.NUM_DRONES,self.IMG_RES[1],self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.AGGR_PHY_STEPS !=0:
                print("[ERROR] in BaseAviary.__init__(), aggregate_phy_steps incompaatible with the desired video capture frame rate ({:f}Hz)").formaat(self.IMG_FRAME_PER_SEC)
                exit()
            if self.RECORD:
                self.ONBOARD_IMG_PATH=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/onboard-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH),exit_ok=True)

        #create attributes for dynamics control inputs
        self.DYNAMICS_ATTR=dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL==DroneModel.CF2X:
                self.A=np.array([[1,1,1,1],[.5,.5,-.5,-.5],[-.5,.5,.5,-.5],[-1,1,-1,1]])
            elif self.DRONE_MODEL in [DroneModel.CF2P,DroneModel.HB]:
                self.A=np.array([[1,1,1,1],[0,1,0,-1],[-1,0,1,0],[-1,1,-1,1]])
            self.INV_A=np.linalg.inv(self.A)
            self.B_COEFF=np.array([1/self.KF,1/(self.KF*self.L),1/(self.KF*self.L),1/self.KM])
        #connect to PyBullet
        if self.GUI:
            print("#with debug GUI")
            self.CLIENT=p.connect(p.GUI) #p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW,p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i,0,physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0,0,0],
                                         physicsClientId=self.CLIENT
                                         )
            ret=p.getDebugVisualizerCamera(physicsClientID=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #Add input sliders to the GUI
                self.SLIDERS=-1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i]=p.addUserDebugParameter("Propeller "+str(i) +" RPM",0,self.MAX_RPM,self.HOVER_RPM,physicsClientId=self.CLIENT)
                self.INPUT_SWITCH=p.addUserDebugParameter("Use GUI RPM",9999,-1,0,physicsClientId=self.CLIENT)
        else:
            print("#without debug GUI")
            self.CLIENT=p.connect(p.DIRECT)
            if self.RECORD:
                #set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC=24
                self.CAPTURE_FREQ=int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW=p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                  yaw=-30,
                                                                  pitch=-30,
                                                                  roll=0,
                                                                  cameraTargetPosition=[0,0,0],
                                                                  upAxisIndex=2,
                                                                  physicsClientId=self.CLIENT)
                self.CAM_PRO=p.computeProjectionMatrixFOV(fov=60.0,
                                                          aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                          nearVal=0.1,
                                                          farVal=1000.0)
        print("#set initial positions")
        if initial_xyzs is None:
            self.INIT_XYZS=np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]),\
                                      np.array([y*4*self.L for y in range(self.NUM_DRONES)]),\
                                      np.ones(self.NUM_DRONES)*(self.COLLISION_H/2 - self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES,3)
        elif np.array(initial_xyzs).shape==(self.NUM_DRONES,3):
            self.INIT_XYZS=initial_xyzs
        else:
            print("[Error] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")

        if initial_rpys is None:
            self.INIT_RPYS=np.zeros((self.NUM_DRONES,3))
        elif np.array(initial_rpys).shape==(self.NUM_DRONES,3):
            self.INIT_RPYS=initial_rpys
        else:
            print("[Error] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")

        #create action and observation spaces
        self.action_space=self._actionSpace()
        self.observation_space=self._observationSpace()

        print("#housekeeping")
        self._housekeeping()

        #update and store the drones kinematic information
        self._updateAndStoreKinematicInformation()

        #Start video recording
        self._startVideoRecording()

    def reset(self):
        """Resets the environment and returns the ndarray or
        dict[..]
        The initial observation, check the specific implementation of '_computeObs()'
        in each subclass for its format.
        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #Housekeeping
        self._housekeeping()
        #update and store the drones kinematic information
        self._updateAndStoreKinematicInformation()
        #start video recording
        self._startVideoRecording()
        #return the initial observation
        return self._computeObs()

    def step(self,action):
        """Advances the environment by one simulation step.
        Parameters
        action: ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by the
            specific implementation of `_preprocessAction()` in each subclass for its
            format.

        Returns
        _______
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.

        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.

        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #save the png video frames if RECORD=True and GUI=False
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ==0:
            [w,h,rgb,dep,seg]=p.getCameraImage(width=self.VID_WIDTH,
                                               height=self.VID_HEIGHT,
                                               shadow=1,
                                               viewMatrix=self.CAM_VIEW,
                                               projectionMatrix=self.CAM_PRO,
                                               renderer=p.ER_TINY_RENDERER,
                                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                               physicsClientId=self.CLIENT
                                               )
            (Image.fromarray(np.reshape(rgb,(h,w,4)),'RGBA')).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM+=1
        #Read the GUI's input parameters
        if self.GUI and self.USER_DEBUG:
            current_input_switch=p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch>self.last_input_switch:
                self.last_input_switch=current_input_switch
                self.USE_GUI_RPM=True if self.USE_GUI_RPM==False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i]=p.readUserDebugParameter(int(self.SLIDERS[i]),physicsClientId=self.CLIENT)
            clipped_action=np.tile(self.gui_input,(self.NUM_DRONES,1))
            if self.step_counter%(self.SIM_FREQ/2)==0:
                self.GUI_INPUT_TEXT=[p.addUserDebugText("Using GUI RPM",
                                                        textPosition=[0,0,0],
                                                        textColorRGB=[1,0,0],
                                                        lifeTime=1,
                                                        textSize=2,
                                                        parentObjectUniqueId=self.DRONE_IDS[i],
                                                        parentLinkIndex=-1,
                                                        replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                        physicsClientId=self.CLIENT
                                                        ) for i in range(self.NUM_DRONES)]
        else:
            self._saveLastAction(action)
            clipped_action=np.reshape(self._preprocessAction(action),(self.NUM_DRONES,4))
        #repeat for as many as the aggregate physics steps
        for _ in range(self.AGGR_PHY_STEPS):
            #update and store the drones kinematic information for certain
            #between aggregate steps for certain types of update
            if  self.AGGR_PHY_STEPS>1 and self.PHYSICS in [Physics.DYN,Physics.PYB_GND,Physics.PYB_DRAG,Physics.PYB_DW,Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            for i in range(self.NUM_DRONES):
                if self.PHYSICS==Physics.PYB:
                    self._physics(clipped_action[i,:],i)
                elif self.PHYSICS==Physics.DYN:
                    self._dynamics(clipped_action[i,:],i)
                elif self.PHYSICS==Physics.PYB_GND:
                    self._physics(clipped_action[i,:],i)
                    self._drag(self.last_clipped_action[i,:],i)
                elif self.PHYSICS==Physics.PYB_DW:
                    self._physics(clipped_action[i,:],i)
                    self._downwash(i)
                elif self.PHYSICS==Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i,:],i)
                    self._groundEffect(clipped_action[i,:],i)
                    self._drag(self.last_clipped_action[i,:],i)
                    self._downwash(i)
            #PyBullet computes the new state unless Physics.DYN
            if self.PHYSICS!=Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #Save the last applied action (e.g to compute drag)
            if self.PHYSICS in [Physics.PYB_DRAG,Physics.PYB_GND_DRAG_DW]:
                self.last_clipped_action=clipped_action
        #update and store the drones kinematic information
        self._updateAndStoreKinematicInformation()
        #prepare the return values
        obs=self._computeObs()
        reward=self._computeReward()
        done=self._computeDone()
        info=self._computeInfo()
        #advance the step counter
        self.step_counter=self.step_counter+(1*self.AGGR_PHY_STEPS)
        return obs,reward,done,info

    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.
        Parameters
        ----------
        mode: str,optional
            Unused.
        close : bool,optional
            Unused.
        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, "
                  "re-initialize the environment using Aviary(qui=True) to use PyBullet's "
                  "graphical interface")
            self.first_render_call=False
        print("\n [INFO] BaseAviary.render() --- it {:04d}".format(self.step_counter),
              "--- wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.TIMESTEP,self.SIM_FREQ,(self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))

        for i in range(self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ---drone {:d}".format(i),
                  "--- x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i,0],self.pos[i,1],self.pos[i,2]),
                  "--- velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i,0],self.vel[i,1],self.vel[i,2]),
                  "--- roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i,0]*self.RAD2DEG,self.rpy[i,1]*self.RAD2DEG,self.rpy[i,2]*self.RAD2DEG),
                  "--- angular velocities {:+06.2f}, {:+06.2f}, {:+06.2f} ---".format(self.ang_v[i,0]*self.RAD2DEG,self.ang_v[i,1]*self.RAD2DEG,self.ang_v[i,2]*self.RAD2DEG)
                  )

    def close(self):
        """Terminates the environment"""
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID,physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.
        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT

    def getDroneIds(self):
        """Return the Drone Ids.
        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.
        """
        return self.DRONE_IDS

    def _housekeeping(self):
        """
        Housekeeping function.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.
        """
        #Initialize/reset counters and zero-valued variables
        self.RESET_TIME=time.time()
        self.step_counter=0
        self.first_render_call=True
        self.X_AX= -1*np.ones(self.NUM_DRONES)
        self.Y_AX= -1*np.ones(self.NUM_DRONES)
        self.Z_AX= -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT= -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch=0
        self.last_action= -1*np.ones((self.NUM_DRONES,4))
        self.last_clipped_action= np.zeros((self.NUM_DRONES,4))
        self.gui_input=np.zeros(4)

        #Initialize the drones kinematic information
        self.pos=np.zeros((self.NUM_DRONES,3))
        self.quat=np.zeros((self.NUM_DRONES,4))
        self.rpy=np.zeros((self.NUM_DRONES,3))
        self.vel=np.zeros((self.NUM_DRONES,3))
        self.ang_v=np.zeros((self.NUM_DRONES,3))

        #Set PyBullet's parameters
        p.setGravity(0,0,-self.G,physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0,physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP,physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=self.CLIENT)

        print("#load ground plane, drone and obstacles models")
        self.PLANE_ID=p.loadURDF("plane.urdf",physicsClientId=self.CLIENT)
        print("meoooooooooooooooouuuuuuuuuuuuuuu")
        # x=(os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + self.URDF)
        # print("path>>",x)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + self.URDF,
                                              self.INIT_XYZS[i, :],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                                              physicsClientId=self.CLIENT) for i in range(self.NUM_DRONES)])
        print("urdf loading.....")
        for i in range(self.NUM_DRONES):
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
                #disable collisions between drones' and ground plane
        if self.OBSTACLES:
            self._addObstacles()

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinematic information.
            This method is meant to limit the number of calls to PyBullet
            in each step and improve performance (at the expense of memory)
        """
        for i in range(self.NUM_DRONES):
            self.pos[i],self.quat[i]=p.getBasePositionAndOrientation(self.DRONE_IDS[i],physicsClientId=self.CLIENT)
            self.rpy[i]=p.getEulerFromQuaternion(self.quat[i])
            self.vel[i],self.ang_v[i]=p.getBaseVelocity(self.DRONE_IDS[i],physicsClientId=self.CLIENT)

    def _startVideoRecording(self):
       """Starts the recording of a video output.
       The format of the video output is .mp4, if GUI is True, or .png, otherwise.
       The video is saved under folder `files/videos`.
       """
       if self.RECORD and self.GUI:
           self.VIDEO_ID=p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                             fileName=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4",
                                             physicsClientId=self.CLIENT
                                             )
       if self.RECORD and not self.GUI:
           self.FRAME_NUM=0
           self.IMG_PATH=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
           os.makedirs(os.path.dirname(self.IMG_PATH),exist_ok=True)

    def _getDroneStateVector(self,nth_drone):
        """Returns the state vector of the nth drone.
            Parameters
            ----------
            nth_drone:int
                The ordinal number/position of the desired drone in list self.DRONE_IDS.

            Returns
            -------
            ndarray
                (20,)-shaped array of floats containing the state vector of the nth drone.
                Check the only line in this method and `_updateAndStoreKinematicInformation()`
                to understand its format.
        """
        state=np.hstack([self.pos[nth_drone,:],self.quat[nth_drone,:],self.rpy[nth_drone,:],
                         self.vel[nth_drone,:],self.ang_v[nth_drone,:],self.last_action[nth_drone,:]])
        return state.reshape(20,)

    def _getDroneImages(self,nth_drone,segmentation: bool=True):
        """Returns camera captures from the nth drone POV.
        Parameters
        ----------
        nth_drone: int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation: bool, optional
            Whether to compute the segmentation mask
            It affects performance

        Returns
        -------
        ndarray
            (h,w,4)-shaped array of uint8's containing the RBG(A) image captured from the nth drone's POV.
        ndarray
            (h,w)-shaped array of uint8's containing the depth image captured from the nth drone's POV.
        ndarray
            (h,w)-shaped array of uint8's containing the segmentation image captured from the nth drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width,height])")
            exit()
        rot_mat=np.array(p.getMatrixFromQuaternion(self.quat[nth_drone,:])).reshape(3,3)

        #set target point, camera view and projection matrices
        target=np.dot(rot_mat,np.array([1000,0,0]))+np.array(self.pos[nth_drone,:])
        DRONE_CAM_VIEW=p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone,:]+np.array([0,0,self.L]),
                                           cameraTargetPosition=target,
                                           cameraUpVector=[0,0,1],
                                           physicsClientId=self.CLIENT
                                           )
        self.dview=DRONE_CAM_VIEW
        DRONE_CAM_PRO=p.computeProjectionMatrixFOV(fov=60.0,
                                                   aspect=1.0,
                                                   nearVal=self.L,
                                                   farVal=1000.0
                                                   )
        SEG_FLAG=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w,h,rgb,dep,seg]=p.getCameraImage( width=self.IMG_RES[0],
                                            height=self.IMG_RES[1],
                                            shadow=1,
                                            viewMatrix=DRONE_CAM_VIEW,
                                            projectionMatrix=DRONE_CAM_PRO,
                                            flags=SEG_FLAG,
                                            physicsClientId=self.CLIENT
                                            )
        rgb=np.reshape(rgb,(h,w,4))
        dep=np.reshape(dep,(h,w))
        seg=np.reshape(seg,(h,w))

        return rgb,dep,seg


    def _exportImage(self,img_type:ImageType,img_input,path:str,frame_num:int=0):
        """Returns camera captures from the nth drone POV
        Parameters
        ----------
        img_type: ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input: ndarray
            (h,w,4)-shaped array of uint8's for RBG(A) or B&W images
            (h,w)-shaped array of uint8's for depth or segmentation images.
        path :str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append the PNG's filename

        """
        if img_type==ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'),'RGBA')).save(path+"frame_"+str(frame_num)+".png")
        elif img_type==ImageType.DEP:
            temp=((img_input-np.min(img_input))*255/(np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type==ImageType.SEG:
            temp=((img_input-np.min(img_input))*255/(np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type==ImageType.BW:
            temp=(np.sum(img_input[:,:,0:2],axis=2)/3).astype('uint8')
        else:
            print("[Error] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type !=ImageType.RGB:
            (Image.fromarray(temp)).save(path+"frame_"+str(frame_num)+".png")


    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of multi-drone system.
            Attribute Neighbourhood_radius is used to determine neighboring relationships.

        """
        adjacency_mat=np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i,:]-self.pos[j+i+1,:])< self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i,j+i+1]=adjacency_mat[j+i+1,i]=1
        return adjacency_mat

    def _physics(self,rpm,nth_drone):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm: ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone: int
            The ordinal number/position of the desired drone in list self.DRONE_IDS
        """
        forces=np.array(rpm**2)*self.KF
        torques=np.array(rpm**2)*self.KM
        z_torque=(-torques[0]+torques[1]-torques[2]+torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0,0,forces[i]],
                                 posObj=[0,0,0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0,0,z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    def _groundEffect(self,rpm,nth_drone):
        """
        PyBullet implementation of a ground effect model.
        :param rpm:
            ndarray
                (4)-shaped array of ints containing the RPMs values of the 4 motors
        :param nth_drone:
            int
                The ordinal number/position of the desired drone in list self.DRONE_IDS

        """
        #Kinematics information of all links(propellers and center of mass)
        link_states=np.array(p.getLinkStates(self.DRONE_IDS[nth_drone],
                                            linkIndices=[0,1,2,3,4],
                                            computeLinkVelocity=1,
                                             computeForwardKinematics=self.CLIENT
                                             ))
        #Simple, per-propeller ground effects
        prop_heights=np.array([link_states[0,0][2],link_states[1,0][2],link_states[2,0][2],link_states[3,0][2]])
        prop_heights=np.clip(prop_heights,self.GND_EFF_H_CLIP,np.inf)
        gnd_effects=np.array(rpm**2)* self.KF * self.GND_EFF_COEFF* (self.PROP_RADIUS/(4*prop_heights))**2
        if np.abs(self.rpy[nth_drone,0])<np.pi/2 and np.abs(self.rpy[nth_drone,1])< np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0,0,gnd_effects[i]],
                                     posObj=[0,0,0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    def _drag(self,rpm,nth_drone):
        """
        PyBullet implementation of a drag model

        :param rpm: ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        :param nth_drone:
            The ordinal number/position of the desired drone in list self.DRONE_IDS
        :return:
        """
        #Rotation matrix of the base
        base_rot=np.array(p.getMatrixFromQuaternion(self.quat[nth_drone,:])).reshape(3,3)
        #simple draft model applied to the base/center of mass
        drag_factors=-1*self.DRAG_COEFF*np.sum(np.array(2*np.pi*rpm/60))
        drag=np.dot(base_rot,drag_factors*np.array(self.vel[nth_drone,:]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0,0,0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
    def _downwash(self,nth_drone):
        """
        PyBullet implementation of a ground effect model
        :param nth_drone: int
            The ordinal number/position of the desired drone in list self.DRONE_IDS
        :return:
        """
        for i in range(self.NUM_DRONES):
            delta_z=self.pos[i,2]-self.pos[nth_drone,2]
            delta_xy=np.linalg.norm(np.array(self.pos[i,0:2])-np.array(self.pos[nth_drone,0:2]))
            if delta_z>0 and delta_xy<10:
                alpha=self.DW_COEFF_1*(self.PROP_RADIUS/(4*delta_z))**2
                beta=self.DW_COEFF_2*delta_z+self.DW_COEFF_3
                downwash=[0,0,-alpha*np.exp(-.5*(delta_xy/beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0,0,0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    def _dynamics(self,rpm,nth_drone):
        """

        :param rpm: ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors
        :param nth_drone: int
            The ordinal number/position of the desired drone in list self.DRONE_IDS
        :return:
        """
        #current state
        pos=self.pos[nth_drone,:]
        quat=self.quat[nth_drone,:]
        rpy=self.rpy[nth_drone,:]
        vel=self.vel[nth_drone,:]
        ang_v=self.ang_v[nth_drone,:]
        rotation=np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
        #compute forces and torques
        forces=np.array(rpm**2)*self.KF
        thrust=np.array([0,0,np.sum(forces)])
        thrust_world_frame=np.dot(rotation,thrust)
        force_world_frame=thrust_world_frame-np.array([0,0,self.GRAVITY])
        z_torques=np.array(rpm**2)*self.KM
        z_torque=(-z_torques[0]+z_torques[1]-z_torques[2]+z_torques[3])
        if self.DRONE_MODEL==DroneModel.CF2X:
            x_torque=(forces[0]+forces[1]-forces[2]-forces[3])*(self.L/np.sqrt(2))
            y_torque=(-forces[0]+forces[1]+forces[2]-forces[3])*(self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P or self.DRONE_MODEL==DroneModel.HB:
            x_torque=(forces[1]-forces[3])*self.L
            y_torque=(-forces[0]+forces[2])*self.L
        torques=np.array([x_torque,y_torque,z_torque])
        torques=torques-np.cross(ang_v,np.dot(self.J,ang_v))
        ang_vel_deriv=np.dot(self.J_inv,torques)
        no_pybullet_dyn_accs=force_world_frame/self.M

        #update state
        vel=vel+self.TIMESTEP*no_pybullet_dyn_accs
        ang_v=ang_v+self.TIMESTEP*ang_vel_deriv
        pos=pos+self.TIMESTEP*vel
        rpy=rpy+self.TIMESTEP*ang_v

        #set PyBullet's state
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.CLIENT
                                          )
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            ang_v,
                            physicsClientId=self.CLIENT
                            )


    def _normalizedActionToRPM(self,action):
        """

        :param action:
            ndarray
            (4)-shaped array of ins containing an input in the [-1,1] range.
        :return:
            ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0,MAXRPM] range
        """
        if np.any(np.abs(action))>1:
            print("\n [ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action<=0, (action+1)*self.HOVER_RPM,action*self.MAX_RPM)

    def _saveLastAction(self,action):
        """
        stores the most recent action into attribute self.last_action
        The last action can be used to compute the aerodynamic effects
        (for single or multi-agent aviaries, respectively)

        :param action:
            ndarray | dict
                (4)-shaped array of ints(or dictionary of arrays) containing the current RPMs input

        :return:
        """
        if isinstance(action,collections.Mapping):
            for k,v in action.items():
                res_v=np.resize(v,(1,4))
                self.last_action[int(k),:]=res_v
        else:
            res_action=np.resize(action,(1,4))
            self.last_action=np.reshape(res_action,(self.NUM_DRONES,4))

    def _showDroneLocalAxes(self,nth_drone):
        """
        Draw the local frame of the nth drone in PyBullet's GUI
        :param nth_drone:
            int
            The ordinal number/position of the desired drone in list self.DRONE_IDs
        :return:
        """
        if self.GUI:
            AXIS_LENGTH=2*self.L
            self.X_AX[nth_drone]=p.addUserDebugLine(lineFromXYZ=[0,0,0],
                                                    lineToXYZ=[AXIS_LENGTH,0,0],
                                                    lineColorRGB=[1,0,0],
                                                    parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                    physicsClientId=self.CLIENT
                                                    )
            self.Y_AX[nth_drone]=p.addUserDebugLine(lineFromXYZ=[0,0,0],
                                                    lineToXYZ=[0,AXIS_LENGTH,0],
                                                    lineColorRGB=[0,1,0],
                                                    parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                    physicsClientId=self.CLIENT
                                                    )
            self.Z_AX[nth_drone]=p.addUserDebugLine(lineFromXYZ=[0,0,0],
                                                    lineToXYZ=[0,0,AXIS_LENGTH],
                                                    lineColorRGB=[0,0,1],
                                                    parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                    physicsClientId=self.CLIENT
                                                    )
    def _addObstacles(self):
        """
        Add obstacles to the environment
        These obstacles are loaded from standard URDF files included in Bullet
        :return:
        """
        # p.loadURDF("sphere2.urdf",
        #            [2,2,.5],
        #            p.getQuaternionFromEuler([0,0,0]),
        #            physicsClientId=self.CLIENT
        #            )

    def _parseURDFParameters(self):
        """
        Loads the parameters from an URDF file.
        This method is like a custom XML parser for the .urdf files in folder `assets/`.
        :return:
        """
        URDF_TREE=etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF).getroot()
        print("drone path: ",os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF)
        M=float(URDF_TREE[1][0][1].attrib['value'])
        L=float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO=float(URDF_TREE[0].attrib['thrust2weight'])
        IXX=float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY=float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ=float(URDF_TREE[1][0][2].attrib['izz'])
        J=np.diag([IXX,IYY,IZZ])
        J_INV=np.linalg.inv(J)
        KF=float(URDF_TREE[0].attrib['kf'])
        KM=float(URDF_TREE[0].attrib['km'])
        COLLISION_H=float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R=float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS=[float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET=COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH=float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF=float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS=float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY=float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z=float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF=np.array([DRAG_COEFF_XY,DRAG_COEFF_XY,DRAG_COEFF_Z])
        DW_COEFF_1=float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2=float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3=float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M,L,THRUST2WEIGHT_RATIO,J,J_INV,KF,KM,COLLISION_H,COLLISION_R,COLLISION_Z_OFFSET,\
        MAX_SPEED_KMH, GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1,DW_COEFF_2,DW_COEFF_3


    def _actionSpace(self):
        """
        Returns the action space of the environment
        Must be implemented in a subclass
        :return:
        """
        raise NotImplementedError

    def _observationSpace(self):
        """
        Returns the observation space of the environment
        Must be implemented in the subclass
        :return:
        """
        raise NotImplementedError

    def _computeObs(self):
        """
        Returns the current observation of the environment.
        Must be implemented in a subclass
        :return:
        """
        raise NotImplementedError
    def _preprocessAction(self,action):
        """
        Preprocess the action passed to `.step()` into motors' RPMs.
        Must be implemented in a subclass
        :param action:
            ndarray | dict[..]
                The input action for one or more drones, to be translated into RPMs.
        :return:
        """
        raise NotImplementedError
    def _computeReward(self):
        """
        Computes the current reward value(s).
        Must bee implemented in a subclass
        :return:
        """
        raise NotImplementedError
    def _computeDone(self):
        """
        Computes the current done value(s).
        Must be implemented in a subclass.
        :return:
        """
        raise NotImplementedError
    def _computeInfo(self):
        """
        Computes the current info dict(s).
        Must be implemeted in a subclass
        :return:
        """

        raise NotImplementedError


