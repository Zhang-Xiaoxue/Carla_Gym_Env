# %%
import os
os.environ["CARLA_ROOT"] = "/opt/carla-simulator"
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import copy
import subprocess
import random
import numpy as np
import time
from skimage.transform import resize

from gym.utils import seeding
import gym
import pygame
from pygame.locals import *
import carla

# from agents.navigation.local_planner import RoadOption
from hud import HUD
from route_planner import RoutePlanner
from render import BirdeyeRender
from wrappers import get_actor_display_name, kill_process
from misc import *

dir_path = os.path.dirname(os.path.realpath(__file__))

# %%
class CARLA_PARAMS:
    """This class is to set the parameters for CARLA simulator.

        To run an agent in this environment, either start start CARLA beforehand with:

        Synchronous:  $> ./CarlaUE4.sh Town* -benchmark -fps=**
        Asynchronous: $> ./CarlaUE4.sh Town*

        Or, pass argument -start_carla in the command-line.
        Note that ${CARLA_ROOT} needs to be set to CARLA's top-level directory
        in order for this option to work.

        And also remember to set the -fps and -synchronous arguments to match the
        command-line arguments of the simulator (not needed with -start_carla.)

        Note that you may also need to add the following line to
        Unreal/CarlaUE4/Config/DefaultGame.ini to have the map included in the package:

        +MapsToCook=(FilePath="/Game/Carla/Maps/Town07")
    """
    host = "127.0.0.1"        # IP address of the CARLA host
    port = 2000               # connection port   
    start_carla = False        # If True, start CARLA from this script. If False, from terminal.
    is_display_interaction = True  # if True, display the images of sensors. If False, no display.

    # Resolution of the spectator camera (placed behind the vehicle by default) as a (viewer_res, viewer_res) tuple
    viewer_res = 768
    # Resolution of the observation camera (placed on the dashboard by default) as a (viewer_res/3, viewer_res/3) tuple
    obs_res = int(viewer_res/3)
    synchronous = False        # If True, run in synchronous mode
    # FPS of the client. If fps <= 0 then use unbounded FPS.
    # Note: Sensors will have a tick rate of fps when fps > 0, otherwise they will tick as fast as possible.
    fps = 30
    map_name = "Town03"       # which town to simulate

    max_past_step = 1  # the number of past steps to draw
    # mode of the task, [random, roundabout (only for Town03)]
    task_mode = "random"
    max_time_episode = 100  # maximum timesteps per episode
    max_ego_spawn_times = 10  # maximum times to spawn ego vehicle
    max_waypt = 20

    number_of_vehicles = 100
    number_of_walkers = 0
    ego_vehicle_filter = 'vehicle.lincoln.mkz2017'  # filter for defining ego vehicle
    desired_speed = 8  # desired speed (m/s)
    out_lane_thres = 2.0  # threshold for out of lane

    # if True, use discrete action space; else, continuous action space
    is_discrete = False
    if is_discrete:
        discrete_steer = [-1.0, 0.0, 1.0]
        discrete_throt = [0.0, 0.5, 1.0]
        discrete_brake = [0.0, 0.5, 1.0]
    else:
        continuous_steer_range = [-1.0, 1.0]  # continuous steering angle range
        continuous_throt_range = [0.0, 1.0]   # continuous throtle range
        continuous_brake_range = [0.0, 1.0]   # continuous brake range

    # Custom reward function that is called every step. If None, no reward function is used.
    reward_fn = None
    # Function that takes the image (of obs_res resolution) from the observation camera and encodes it to some state vector to returned by step(). If None, step() returns the full image.
    encode_state_fn = None
    # Scalar used to smooth the incomming action signal. 1.0 = max smoothing, 0.0 = no smoothing
    action_smoothing = 0.9

    # set the observation channel, including "camera_rgb", "birdeye",
    observation_channel = ["camera_rgb", "birdeye", "camera_seg", "lidar", "camera_dep"]
    display_route_in_birdeye = True
    # parameter setting for lidar
    d_behind = 12  # distance behind the ego vehicle (meter)
    lidar_height = 1.2
    lidar_obs_range = 32 # lidar observation range (meter)
    lidar_bin = lidar_obs_range/obs_res  # bin size of lidar sensor (meter)
    camera_pos = (1.4, 1.7)
    display_route_in_lidar = False
    if ~is_display_interaction:
        display_route_in_lidar = False


    
# %%
class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self):
        self.sensors_list = []
        self.pixor = False
        # Start CARLA from CARLA_ROOT
        self.carla_process = None
        if CARLA_PARAMS.start_carla:
            kill_process()
            launch_command = [os.environ['CARLA_ROOT'] + "/CarlaUE4.sh"]
            # launch_command += ["Game/Carla/Maps/Town07"]
            if CARLA_PARAMS.synchronous:
                launch_command += ["-benchmark -fps=%i" % CARLA_PARAMS.fps]
            launch_command = ' '.join(launch_command)
            print("Running command:", launch_command)
            self.carla_process = subprocess.Popen(
                launch_command, stdout=subprocess.PIPE, universal_newlines=True)
            print("Waiting for CARLA to initialize")
            time.sleep(10.0)

        # set destination
        if CARLA_PARAMS.task_mode == 'roundabout':
            self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0],
                          [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
            self.dests = None

        # Action space
        if CARLA_PARAMS.is_discrete:
            self.discrete_act = [CARLA_PARAMS.discrete_steer,
                                 CARLA_PARAMS.discrete_throt,
                                 CARLA_PARAMS.discrete_brake]  # steer, throtle, brake
            self.action_space = gym.spaces.Discrete(len(CARLA_PARAMS.discrete_steer) *
                                                    len(CARLA_PARAMS.discrete_throt) *
                                                    len(CARLA_PARAMS.discrete_brake))
        else:
            self.action_space = gym.spaces.Box(np.array([CARLA_PARAMS.continuous_steer_range[0],
                                                         CARLA_PARAMS.continuous_throt_range[0],
                                                         CARLA_PARAMS.continuous_brake_range[0]]),
                                               np.array([CARLA_PARAMS.continuous_steer_range[1],
                                                         CARLA_PARAMS.continuous_throt_range[1],
                                                         CARLA_PARAMS.continuous_brake_range[1]]),
                                               dtype=np.float32)  # steer, throttle

        # Observation space
        observation_space_dict = {
            'camera_rgb':  gym.spaces.Box(low=0, high=255, shape=(CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8),            
        }
        if "birdeye" in CARLA_PARAMS.observation_channel:
            observation_space_dict.update({'birdeye':     gym.spaces.Box(
                low=0, high=255, shape=(CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)})
        if "lidar" in CARLA_PARAMS.observation_channel:
            observation_space_dict.update({'lidar':        gym.spaces.Box(
                low=0, high=255, shape=(CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)})
        if "camera_seg" in CARLA_PARAMS.observation_channel:
            observation_space_dict.update({'camera_seg':  gym.spaces.Box(
                low=0, high=255, shape=(CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)})
        if "camera_dep" in CARLA_PARAMS.observation_channel:
            observation_space_dict.update({'camera_dep': gym.spaces.Box(
                low=0, high=255, shape=(CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)})
        self.observation_space = gym.spaces.Dict(observation_space_dict)

        self.world = None
        try:
            # Connect to carla
            print('connecting to Carla server...')
            self.client = carla.Client(CARLA_PARAMS.host, CARLA_PARAMS.port)
            self.client.set_timeout(60.0)
            # Create world wrapper
            self.world = self.client.load_world(CARLA_PARAMS.map_name)
            # Set weather
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            print('Carla server connected!')

            # Get spawn points for other vehicles and walkers
            self.vehicle_spawn_points = list(
                self.world.get_map().get_spawn_points())
            self.walker_spawn_points = []
            for _ in range(CARLA_PARAMS.number_of_walkers):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    self.walker_spawn_points.append(spawn_point)

            # Create the ego vehicle blueprint
            self.ego_bp = self._create_vehicle_bluepprint(
                CARLA_PARAMS.ego_vehicle_filter, color='49,8,8')
            
            sensor_tick = 0.0 if CARLA_PARAMS.synchronous else 1.0/CARLA_PARAMS.fps

            # Collision sensor
            self.collision_hist = []  # The collision history
            self.collision_hist_l = 1  # collision history length
            self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

            # lane invasion senso
            self.lane_invasion_hist = []  # The collision history
            self.lane_invasion_hist_l = 1  # collision history length
            self.lane_invasion_bp = self.world.get_blueprint_library().find(
                'sensor.other.lane_invasion')
            
            # Lidar sensor
            if "lidar" in CARLA_PARAMS.observation_channel:
                self.lidar_data = np.zeros((CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3))
                self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=CARLA_PARAMS.lidar_height))
                self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
                self.lidar_bp.set_attribute('channels', '64')
                self.lidar_bp.set_attribute('range', '50')
                self.lidar_bp.set_attribute('sensor_tick', str(sensor_tick))

            # Viewer
            self.viewer_img = np.zeros(
                (CARLA_PARAMS.viewer_res, CARLA_PARAMS.viewer_res, 3), dtype=np.uint8)
            self.viewer_trans = carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
            self.viewer_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.viewer_bp.set_attribute(
                'image_size_x', str(CARLA_PARAMS.viewer_res))
            self.viewer_bp.set_attribute(
                'image_size_y', str(CARLA_PARAMS.viewer_res))
            self.viewer_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.viewer_bp.set_attribute('sensor_tick', str(sensor_tick))

            # Front Camera RGB sensor
            self.camera_rgb_img = np.zeros(
                (CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)
            self.camera_rgb_trans = carla.Transform(
                carla.Location(x=CARLA_PARAMS.camera_pos[0], z=CARLA_PARAMS.camera_pos[1]))
            self.camera_rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_rgb_bp.set_attribute(
                'image_size_x', str(CARLA_PARAMS.obs_res))
            self.camera_rgb_bp.set_attribute(
                'image_size_y', str(CARLA_PARAMS.obs_res))
            self.camera_rgb_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_rgb_bp.set_attribute('sensor_tick', str(sensor_tick))

            if "camera_seg" in CARLA_PARAMS.observation_channel:
                # Segmentation Camera sensor
                self.camera_seg_img = np.zeros(
                    (CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)
                self.camera_seg_trans = carla.Transform(
                    carla.Location(x=CARLA_PARAMS.camera_pos[0], z=CARLA_PARAMS.camera_pos[1]))
                self.camera_seg_bp = self.world.get_blueprint_library().find(
                    'sensor.camera.semantic_segmentation')
                # Modify the attributes of the blueprint to set image resolution and field of view.
                self.camera_seg_bp.set_attribute(
                    'image_size_x', str(CARLA_PARAMS.obs_res))
                self.camera_seg_bp.set_attribute(
                    'image_size_y', str(CARLA_PARAMS.obs_res))
                self.camera_seg_bp.set_attribute('fov', '110')
                # Set the time in seconds between sensor captures
                self.camera_seg_bp.set_attribute('sensor_tick', str(sensor_tick))
            
            if "camera_dep" in CARLA_PARAMS.observation_channel:
                # Depth Camera sensor
                self.camera_dep_img = np.zeros(
                    (CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res, 3), dtype=np.uint8)
                self.camera_dep_trans = carla.Transform(
                    carla.Location(x=CARLA_PARAMS.camera_pos[0], z=CARLA_PARAMS.camera_pos[1]))
                self.camera_dep_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
                # Modify the attributes of the blueprint to set image resolution and field of view.
                self.camera_dep_bp.set_attribute(
                    'image_size_x', str(CARLA_PARAMS.obs_res))
                self.camera_dep_bp.set_attribute(
                    'image_size_y', str(CARLA_PARAMS.obs_res))
                self.camera_dep_bp.set_attribute('fov', '110')
                # Set the time in seconds between sensor captures
                self.camera_dep_bp.set_attribute('sensor_tick', str(sensor_tick))

            if CARLA_PARAMS.synchronous:
                # Set fixed simulation step for synchronous mode
                self.settings = self.world.get_settings()
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 1/CARLA_PARAMS.fps
                self.world.apply_settings(self.settings)

            # Record the time of total steps and resetting steps
            self.reset_step = 0
            self.total_step = 0

            # Initialize the renderer
            self._init_renderer()

        except Exception as e:
            self.close()
            raise e

    def reset(self):
        self.extra_info = []
        self.total_reward = 0.0

        # Clear sensor objects, vehicles and walkers
        for sensor in self.sensors_list:
            sensor.destroy()
        self.sensors_list = []
        actors_list = ['vehicle.*', 'controller.ai.walker', 'walker.*']
        self._clear_all_actors(actors_list)

        # Disable sync mode
        self._set_synchronous_mode()

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = CARLA_PARAMS.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1


        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = CARLA_PARAMS.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > CARLA_PARAMS.max_ego_spawn_times:
                self.reset()

            if CARLA_PARAMS.task_mode == 'random':
                transform = random.choice(self.vehicle_spawn_points)
            if CARLA_PARAMS.task_mode == 'roundabout':
                self.start = [
                    52.1+np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)
        if CARLA_PARAMS.is_display_interaction:
            self._create_hud()
                
        # Add lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(
            self.lane_invasion_bp, carla.Transform(), attach_to=self.ego)
        self.lane_invasion_sensor.listen(lambda event: get_lane_invasion_hist(event))
        def get_lane_invasion_hist(event):
            lane_types = set(x.type for x in event.crossed_lane_markings)
            text = ["%r" % str(x).split()[-1] for x in lane_types]
            if CARLA_PARAMS.is_display_interaction:
                self.hud.notification("Crossed line %s" % " and ".join(text))           
            self.lane_invasion_hist.append(lane_types)
            if len(self.lane_invasion_hist) > self.lane_invasion_hist_l:
                self.lane_invasion_hist.pop(0)
        self.lane_invasion_hist = []
        self.sensors_list.append(self.lane_invasion_sensor)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            if CARLA_PARAMS.is_display_interaction:
                self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []
        self.sensors_list.append(self.collision_sensor)

        # Add Viewer Spectators
        self.viewer_sensor = self.world.spawn_actor(
            self.viewer_bp, self.viewer_trans, attach_to=self.ego)
        self.viewer_sensor.listen(lambda data: get_viewer_img(data))
        def get_viewer_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.viewer_img = array
        print("Spawned actor \"{}\"".format(self.viewer_sensor.type_id))
        self.sensors_list.append(self.viewer_sensor)
        
        # Add front camera RGB sensor
        self.camera_rgb_sensor = self.world.spawn_actor(
            self.camera_rgb_bp, self.camera_rgb_trans, attach_to=self.ego)
        self.camera_rgb_sensor.listen(lambda data: get_camera_rgb_img(data))
        def get_camera_rgb_img(data):
            # data.save_to_disk(dir_path+"/tmp/rgb_%06d.png" %data.frame, carla.ColorConverter.Raw)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_rgb_img = array
        print("Spawned actor \"{}\"".format(self.camera_rgb_sensor.type_id))
        self.sensors_list.append(self.camera_rgb_sensor)

        if "lidar" in CARLA_PARAMS.observation_channel:
            # Add lidar sensor
            self.lidar_sensor = self.world.spawn_actor(
                self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))
            def get_lidar_data(data):
                self.lidar_data = data
            print("Spawned actor \"{}\"".format(self.lidar_sensor.type_id))
            self.sensors_list.append(self.lidar_sensor)  
        
        if "camera_seg" in CARLA_PARAMS.observation_channel:
            # Add camera segmentation sensor
            self.camera_seg_sensor = self.world.spawn_actor(
                self.camera_seg_bp, self.camera_seg_trans, attach_to=self.ego)
            self.camera_seg_sensor.listen(lambda data: get_camera_seg_img(data))
            def get_camera_seg_img(data): 
                data.convert(carla.ColorConverter.CityScapesPalette)        
                # data.save_to_disk(dir_path+"/tmp/seg_%06d.png" %data.frame, carla.ColorConverter.CityScapesPalette)   
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_seg_img = array
            print("Spawned actor \"{}\"".format(self.camera_seg_sensor.type_id))
            self.sensors_list.append(self.camera_seg_sensor)
        
        if "camera_dep" in CARLA_PARAMS.observation_channel:
            # Add camera depth sensor
            self.camera_dep_sensor = self.world.spawn_actor(
                self.camera_dep_bp, self.camera_dep_trans, attach_to=self.ego)
            self.camera_dep_sensor.listen(lambda data: get_camera_dep_img(data))
            def get_camera_dep_img(data):   
                data.convert(carla.ColorConverter.LogarithmicDepth)  
                # data.save_to_disk(dir_path+"/tmp/dep_%06d.png" %data.frame, carla.ColorConverter.LogarithmicDepth)
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_dep_img = array
            print("Spawned actor \"{}\"".format(self.camera_dep_sensor.type_id))
            self.sensors_list.append(self.camera_dep_sensor)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        self.routeplanner = RoutePlanner(self.ego, CARLA_PARAMS.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        if CARLA_PARAMS.is_display_interaction:
            # Set ego information for render
            self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs()

    def step(self, action):
        # Calculate acceleration and steering
        if CARLA_PARAMS.is_discrete:
            pass # TODO
        else:
            steer, throttle, brake = [float(a) for a in action]

        # Apply control
        act = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(act)

        if CARLA_PARAMS.is_display_interaction:
            self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > CARLA_PARAMS.max_past_step:
            self.vehicle_polygons.pop(0)
            walker_poly_dict = self._get_actor_polygons('walker.*')
            self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > CARLA_PARAMS.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        self.last_reward = self._get_reward()
        self.total_reward += self.last_reward

        return (self._get_obs(), self.last_reward, self._terminal(), copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
        actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
        bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + \
                [x for x in blueprints if int(
                    x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the display viewer and birdeye view.
        """
        if CARLA_PARAMS.is_display_interaction:
            pygame.init()
            pygame.font.init()

            if len(CARLA_PARAMS.observation_channel) > 4:
                self.display = pygame.display.set_mode(
                    (int(1.67*CARLA_PARAMS.viewer_res), CARLA_PARAMS.viewer_res), pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self.display = pygame.display.set_mode(
                    (int(1.34*CARLA_PARAMS.viewer_res), CARLA_PARAMS.viewer_res), pygame.HWSURFACE | pygame.DOUBLEBUF)        
            self.clock = pygame.time.Clock()

            pixels_per_meter = CARLA_PARAMS.obs_res/CARLA_PARAMS.lidar_obs_range
            pixels_ahead_vehicle = (CARLA_PARAMS.lidar_obs_range/2 - CARLA_PARAMS.d_behind) * pixels_per_meter
            birdeye_params = {
                    'screen_size': [CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res],
                    'pixels_per_meter': pixels_per_meter,
                    'pixels_ahead_vehicle':  pixels_ahead_vehicle
                }
            self.birdeye_render = BirdeyeRender(self.world, birdeye_params)
        
    def _set_synchronous_mode(self):
        """Set whether to use the synchronous mode.
        """
        if CARLA_PARAMS.synchronous:
            self.settings.synchronous_mode = True
            self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            'vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(
            self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation())
            # random max speed
            # max speed between 1 and 2 (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
            return True
        return False

    def _create_hud(self):
        """try to create the hud display on the ego vehicle
        """
        self.hud = HUD(CARLA_PARAMS.viewer_res, CARLA_PARAMS.viewer_res)
        self.hud.set_vehicle(self.ego)
        self.world.on_tick(self.hud.on_world_tick)
        
    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args: 
            transform: the carla transform object.
        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
        filt: the filter indicating what type of actors we'll look at.

        Returns:
        actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw/180*np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + \
                np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        if CARLA_PARAMS.is_display_interaction:
            # Birdeye rendering
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # birdeye view with roadmap and actors
            birdeye_pos = (CARLA_PARAMS.viewer_res, CARLA_PARAMS.obs_res)
            birdeye_render_types = ['roadmap', 'actors']
            if CARLA_PARAMS.display_route_in_birdeye:
                birdeye_render_types.append('waypoints')
            if len(CARLA_PARAMS.observation_channel)>3:
                birdeye_center = (self.display.get_width()-self.display.get_height()/2, self.display.get_height()/2)
            else:
                birdeye_center = (self.display.get_width()-CARLA_PARAMS.obs_res, self.display.get_height()/2)
            self.birdeye_render.render(self.display, birdeye_center, birdeye_render_types)
            birdeye = pygame.surfarray.array3d(self.display)
            birdeye = birdeye[CARLA_PARAMS.viewer_res:CARLA_PARAMS.viewer_res+CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res:2*CARLA_PARAMS.obs_res, :]
            birdeye = display_to_rgb(birdeye, CARLA_PARAMS.obs_res)
            if CARLA_PARAMS.is_display_interaction:
                # Display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, CARLA_PARAMS.obs_res)
                self.display.blit(birdeye_surface, birdeye_pos)

        # Display camera image
        camera_rgb = resize(self.camera_rgb_img, (CARLA_PARAMS.obs_res,
                        CARLA_PARAMS.obs_res)) * 255        
        if CARLA_PARAMS.is_display_interaction:
            camera_rgb_surface = rgb_to_display_surface(camera_rgb, CARLA_PARAMS.obs_res)
            self.display.blit(camera_rgb_surface, (CARLA_PARAMS.viewer_res, 0))

        # Blit image from spectator camera
        viewer = resize(self.viewer_img, (CARLA_PARAMS.viewer_res,
                        CARLA_PARAMS.viewer_res)) * 255
        if CARLA_PARAMS.is_display_interaction:
            viewer_surface = rgb_to_display_surface(viewer, CARLA_PARAMS.viewer_res)
            self.display.blit(viewer_surface, (0, 0))

        # Display camera segmentation image
        if "camera_seg" in CARLA_PARAMS.observation_channel:
            camera_seg = resize(self.camera_seg_img, (CARLA_PARAMS.obs_res,
                            CARLA_PARAMS.obs_res)) * 255
            if CARLA_PARAMS.is_display_interaction:
                camera_seg_surface = rgb_to_display_surface(camera_seg, CARLA_PARAMS.obs_res)
                self.display.blit(camera_seg_surface, (CARLA_PARAMS.viewer_res+CARLA_PARAMS.obs_res, 0))

        # Display camera depth image
        if "camera_dep" in CARLA_PARAMS.observation_channel:
            camera_dep = resize(self.camera_dep_img, (CARLA_PARAMS.obs_res,
                            CARLA_PARAMS.obs_res)) * 255
            if CARLA_PARAMS.is_display_interaction:
                camera_dep_surface = rgb_to_display_surface(camera_dep, CARLA_PARAMS.obs_res)
                self.display.blit(camera_dep_surface, (CARLA_PARAMS.viewer_res+CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res))
        
        # Display Lidar image
        if "lidar" in CARLA_PARAMS.observation_channel:
            # Lidar image generation
            point_cloud = []
            # Get point cloud data
            for location in self.lidar_data:
                point_cloud.append(
                    [location.point.x, location.point.y, -location.point.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to CARLA_PARAMS.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(CARLA_PARAMS.lidar_obs_range - CARLA_PARAMS.d_behind),
                            CARLA_PARAMS.d_behind+CARLA_PARAMS.lidar_bin, CARLA_PARAMS.lidar_bin)
            x_bins = np.arange(-CARLA_PARAMS.lidar_obs_range/2, CARLA_PARAMS.lidar_obs_range/2+CARLA_PARAMS.lidar_bin, CARLA_PARAMS.lidar_bin)
            z_bins = [-CARLA_PARAMS.lidar_height-1, -CARLA_PARAMS.lidar_height+0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
            lidar = np.flipud(lidar)
            # Add the waypoints to lidar image
            if CARLA_PARAMS.display_route_in_lidar:
                wayptimg = (birdeye[:, :, 0] <= 10) * \
                    (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            else:
                # wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                wayptimg = np.zeros((CARLA_PARAMS.obs_res, CARLA_PARAMS.obs_res))
            wayptimg = np.roll(wayptimg, int( 3.5*CARLA_PARAMS.obs_res/CARLA_PARAMS.lidar_obs_range), axis=1)
            wayptimg = np.roll(wayptimg, int(-4*CARLA_PARAMS.obs_res/CARLA_PARAMS.lidar_obs_range), axis=0)
            wayptimg[:-30,:] = 0
            wayptimg = np.expand_dims(wayptimg, axis=2)
            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = lidar * 255
            if CARLA_PARAMS.is_display_interaction:
                # Display lidar image
                lidar_surface = rgb_to_display_surface(lidar, CARLA_PARAMS.obs_res)
                self.display.blit(lidar_surface, (CARLA_PARAMS.viewer_res, 2*CARLA_PARAMS.obs_res))
        
        if CARLA_PARAMS.is_display_interaction:
            # display hud infomation  
            self.extra_info.extend([
                "test: ",
                "step_count: %7.2f " % self.time_step,
                # "Reward: % 19.2f" % self.last_reward,
                # "",
                # "Maneuver:        % 11s"       % maneuver,
                # "Distance traveled: % 7d m"    % self.distance_traveled,
                # "Center deviance:   % 7.2f m"  % self.distance_from_center,
                # "Avg center dev:    % 7.2f m"  % (self.center_lane_deviation / self.step_count),
                # "Avg speed:      % 7.2f km/h"  % (3.6 * self.speed_accum / self.step_count) 
                ])      
            self.hud.render(self.display, extra_info=self.extra_info)
            self.extra_info = [] # Reset extra info list

            # Display on pygame
            pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
                                       np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

        obs = {
            'camera_rgb': camera_rgb.astype(np.uint8),
        }

        if CARLA_PARAMS.is_display_interaction:
            if 'birdeye' in CARLA_PARAMS.observation_channel:
                obs.update({'birdeye': birdeye.astype(np.uint8),})
        
        if "lidar" in CARLA_PARAMS.observation_channel:
            obs.update({'lidar': lidar.astype(np.uint8),})
        
        if "camera_dep" in CARLA_PARAMS.observation_channel:
            obs.update({'camera_dep': camera_dep.astype(np.uint8),})
        
        if "camera_seg" in CARLA_PARAMS.observation_channel:
            obs.update({'camera_seg': camera_seg.astype(np.uint8),})

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - CARLA_PARAMS.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > CARLA_PARAMS.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > CARLA_PARAMS.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 200*r_collision + 1*lspeed_lon + 10 * \
            r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > CARLA_PARAMS.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > CARLA_PARAMS.out_lane_thres:
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

# %%
if __name__ == '__main__':
    env = CarlaEnv()
    obs = env.reset()
    while True:
        action = [0.1, 0.2, 0.0]
        obs,r,done,info = env.step(action)

        if done:
            print("done!")
            obs = env.reset()
        
# %%
