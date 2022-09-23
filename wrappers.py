import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame
import psutil
from skimage.transform import resize


def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll
        )
    )

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate-1] + u"\u2026") if len(name) > truncate else name

def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi: angle -= 2 * np.pi
    elif angle <= -np.pi: angle += 2 * np.pi
    return angle

def distance_to_line(A, B, p):
    num   = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom

def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

def kill_process():
    for process in psutil.process_iter(): # kill the existing Carla process
        if process.name().lower().startswith('CarlaUE4'.lower()):
            try: 
                process.terminate()
            except:
                pass
    still_alive = []
    for process in psutil.process_iter():
        if process.name().lower().startswith('CarlaUE4'.lower()):
            still_alive.append(process)
    
    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

CAMERA_TRANSFORMS = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),
    "lidar_pos": carla.Transform(carla.Location(x=0.0, z=2.1))
}

#===============================================================================
# CarlaActorBase
#===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)

#===============================================================================
# CollisionSensor
#===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


#===============================================================================
# LaneInvasionSensor
#===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)

#===============================================================================
# Lidar
#===============================================================================

class Lidar(CarlaActorBase):
    def __init__(self, world, obs_res, transform=carla.Transform(),
                 lidar_params = None,
                 attach_to=None, on_recv_info=None,
                 sensor_type="sensor.lidar.ray_cast"):
        self.on_recv_info = on_recv_info
        self.lidar_params = lidar_params
        self.obs_res = obs_res
        # Setup camera blueprint
        lidar_bp = world.get_blueprint_library().find(sensor_type)
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '500')

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(lidar_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda data: Lidar.process_lidar_input(weak_self, data))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)
    
    @staticmethod
    def process_lidar_input(weak_self, data):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_info):
            # Lidar image generation
            point_cloud = []
            # Get point cloud data
            for location in data:
                point_cloud.append([location.point.x, location.point.y, -location.point.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to CARLA_PARAMS.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(self.lidar_params['lidar_obs_range'] - self.lidar_params['d_behind']),
                                 self.lidar_params['d_behind']+self.lidar_params['lidar_bin'], 
                                 self.lidar_params['lidar_bin'])
            x_bins = np.arange( -self.lidar_params['lidar_obs_range']/2, 
                                 self.lidar_params['lidar_obs_range']/2+self.lidar_params['lidar_bin'], 
                                 self.lidar_params['lidar_bin'])
            z_bins = [-self.lidar_params['lidar_height']-1, -self.lidar_params['lidar_height']+0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
            lidar = np.flipud(lidar)
            # # Add the waypoints to lidar image
            # if CARLA_PARAMS.display_route_in_lidar:
            #     wayptimg = (birdeye[:, :, 0] <= 10) * \
            #         (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            # else:
            #     # wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
            wayptimg = np.zeros((self.obs_res, self.obs_res))
            wayptimg = np.roll(wayptimg, int( 3.5*self.obs_res/self.lidar_params['lidar_obs_range']), axis=1)
            wayptimg = np.roll(wayptimg, int(  -4*self.obs_res/self.lidar_params['lidar_obs_range']), axis=0)
            wayptimg[:-30,:] = 0
            wayptimg = np.expand_dims(wayptimg, axis=2)
            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = lidar * 255
            self.on_recv_info(lidar)

    def destroy(self):
        super().destroy()


#===============================================================================
# Camera
#===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, obs_res, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter
        self.obs_res = obs_res

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(obs_res))
        camera_bp.set_attribute("image_size_y", str(obs_res))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))
        camera_bp.set_attribute('fov', '110')

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)
    
    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = resize(array, (self.obs_res, self.obs_res)) * 255
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()

#===============================================================================
# Vehicle
#===============================================================================

class Vehicle(CarlaActorBase):
    """ spawn vehicle equipped with the collision sensor and lane invansion sensor.  """
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz2017"):
        # Setup vehicle blueprint
        self.vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = self.vehicle_bp.get_attribute("color").recommended_values[0]
        self.vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(self.vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))
            
        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

#===============================================================================
# World
#===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)



