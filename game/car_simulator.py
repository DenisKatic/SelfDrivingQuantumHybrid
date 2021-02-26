#import airsim
import io
#from .deeplearning.utils import *
import numpy

import sys

from airsim import *


class CarSimulator():
    car_controls = None

    # Initialisation of the connection with the Unreal Engine 4 and AirSim simulation
    def __init__(self, ip="127.0.0.1", port=41451):
        self.client = CarClient(ip, port)

    def connect(self):
        self.client.confirmConnection()

    def close(self):
        print("Closing")
        #self.client.client.close()

    def init_control(self):
        self.car_controls = CarControls()

    def enable_api_control(self):
        self.client.enableApiControl(True)
        self.car_controls = CarControls()

    def disable_api_control(self):
        self.client.enableApiControl(False)

    def get_previous_car_controls(self):
        try:
            # New Version
            return self.client.getCarControls()
        except:
            return CarControls()

    def get_car_controls(self):
        return self.car_controls

    def set_new_car_controls(self, new_car_controls):
        self.client.setCarControls(new_car_controls)

    def get_images(self, camera_name, image_type, roi=None):
        try:
            responses = self.client.simGetImages([ImageRequest(camera_name, image_type, False, False)])
            image_response = responses[0]
            image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
            image_rgb = image1d.reshape(image_response.height, image_response.width, 3)

            """
            tmp_img = Image.fromarray(image1d)
            tmp_img.crop((0, image_response.height, ))
            tmp_img.save(file_path)
            """
            # expected input_9 to have shape (None, 59, 255, 3)
            if roi:
                # 255 for rescale
                return image_rgb[roi[0]:roi[1], roi[2]:roi[3], 0:3].astype(float)/255
            else:
                return image_rgb
            """
            image_response = self.client.simGetImage(camera_name, image_type)
            img = Image.open(io.BytesIO(image_response))
    
            # img.save('pil_red.png')
    
            returnIm = process_image_to_np_array(img)
            return returnIm
            """
        except ValueError:
            return []

    def get_car_state(self):
        car_state = self.client.getCarState()
        return car_state

    def get_collision_info(self):
        return self.client.simGetCollisionInfo()

    def reset_car(self):
        self.client.reset()

