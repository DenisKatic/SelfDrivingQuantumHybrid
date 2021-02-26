
from airsim import *
from game.car_simulator import CarSimulator
from ml.model import DeepLearningModel
from ml.additional.preprocessing import *
from multiprocessing import Queue, Manager, Lock
import datetime
import glob
import matplotlib.pyplot as plt


class Environment():
    ai_sim_car = None
    stop_driving = False
    lock = Lock()
    lap = None
    deep_learning_model = None
    ENVIRONMENT_PARAM_KEY = "env_params"
    MODEL_PARAM_KEY = "model_params"
    model_params = None

    # default values
    env_params = DictAttr(
        database_name="ai_driving",
        ai_car_sim_ip="127.0.0.3",
        ai_car_sim_port=41451,

        model_training_folder="./images/raw_data/",
        model_processed_folder="./images/preprocessed/",
        model_output_folder="./models/",
        data_batch_size=32,

        # Model params
        # model_image_shape = (1, 144, 256, 3),
        model_car_stage_shape=(1, 4),
        region_of_interest=[76, 135, 0, 255]
    )
    trained_rotations = None
    hybrid_type = 0

    def init_with_new_values(self, values):
        if self.ENVIRONMENT_PARAM_KEY in values:
            for key in values[self.ENVIRONMENT_PARAM_KEY]:
                if key in self.env_params:
                    self.env_params[key] = values[self.ENVIRONMENT_PARAM_KEY][key]
        if self.MODEL_PARAM_KEY in values:
            self.model_params = values[self.MODEL_PARAM_KEY]

    def add_hybrid_type(self, hybrid_type):
        self.hybrid_type = hybrid_type

    def add_qml_rotations(self, qml_rotations):
        self.trained_rotations = qml_rotations

    def init_ai_car_sim(self):
        self.ai_sim_car = CarSimulator(self.env_params.ai_car_sim_ip, self.env_params.ai_car_sim_port)

    def connect_to_ai_sim(self):
        self.ai_sim_car.connect()

    def reset_ai_car(self):
        self.ai_sim_car.reset_car()

    def prepare_deep_learning_model(self, model_name=None, without_model=False):
        if not self.deep_learning_model:
            self.deep_learning_model = DeepLearningModel(self.env_params.model_processed_folder,
                                                         self.env_params.model_output_folder,
                                                         self.env_params.data_batch_size,
                                                         self.trained_rotations, self.hybrid_type)
            if self.model_params:
                self.deep_learning_model.change_default_params(self.model_params)
            if not without_model:
                if model_name:
                    self.deep_learning_model.load_model(model_name)
        else:
            if not without_model:
                if model_name:
                    self.deep_learning_model.load_model(model_name)

            else:
                self.deep_learning_model.reset_model()

    def start_ai_track(self):
        print("Start ai track")
        self.init_ai_car_sim()
        self.ai_sim_car.connect()
        self.ai_sim_car.enable_api_control()
        self.ai_sim_car.reset_car()

        self.stop_driving = False

        shape_img_buf, region_of_interest = self.deep_learning_model.get_image_buf_shape_and_roi()
        image_buf = np.zeros(shape_img_buf)
        state_buf = np.zeros(self.env_params.model_car_stage_shape)


        index = 0
        save_current=False # this flag should save each second timestamp
        crash_count = 0
        start = datetime.datetime.now()
        while not self.stop_driving:
            index = index +1
            if self.ai_sim_car.get_collision_info().has_collided:
                if crash_count > 1:
                    self.stop_driving = True
                    break
                else:
                    crash_count = crash_count +1


            car_state = self.ai_sim_car.get_car_state()
            img1d = self.ai_sim_car.get_images("0", ImageType.Scene, region_of_interest)
            if len(img1d) > 0:

                #prev_car_controls = self.ai_sim_car.get_previous_car_controls()

                car_controls = self.ai_sim_car.get_car_controls()
                image_buf[0] = img1d
                state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])

                result = self.deep_learning_model.predict_result(image_buf, state_buf)
                predicted_steering = result[0][0]
                new_steering = round(float(predicted_steering), 2)
                new_steering = new_steering
                #print("Predicted: " + str(predicted_steering) + " post_process:" + str(new_steering))

                car_controls.steering = new_steering

                # Additional car stats
                new_throttle = 0
                if -0.1 < car_controls.steering < 0.1:
                    if car_state.speed < 4:
                        new_throttle = 1.0
                    else:
                        new_throttle = 0.0
                else:
                    if car_state.speed < 4:
                        new_throttle = 1.0
                    else:
                        new_throttle = 0.0
                car_controls.throttle = new_throttle

                self.ai_sim_car.set_new_car_controls(car_controls)
            else:
                # No picture
                print("No picture!")
                prev_car_controls = self.ai_sim_car.get_previous_car_controls()
                self.ai_sim_car.set_new_car_controls(prev_car_controls)


        print("Count iterations: " + str(index))
        print("Total time:" + str(round((datetime.datetime.now() - start).total_seconds(), 2)))
        self.ai_sim_car.disable_api_control()
        self.ai_sim_car.reset_car()
        self.ai_sim_car.close()

    def start_training_ai(self, training_folders=None, epoch=None, model_name=None):
        pre_processing = None

        if training_folders:
            pre_processing = PreProcess()
            pre_processing.prepare_training_data(training_folders, self.env_params.model_processed_folder,
                                        self.env_params.data_batch_size)
        else:
            pre_processing = PreProcess()
            pre_processing.prepare_training_data(self.env_params.model_training_folder, self.env_params.model_processed_folder,
                                        self.env_params.data_batch_size)
        pre_processing.start_processing()

        if model_name:
            self.prepare_deep_learning_model(model_name)
        else:
            self.prepare_deep_learning_model(without_model=True)
        self.deep_learning_model.start_training(epoch)

    def start_test(self, test_folder, original_model=None):
        if not test_folder:
            raise Exception("No test folder defined!")
        print("Start test")

        classic_model = None
        if original_model:
            classic_model = DeepLearningModel(self.env_params.model_processed_folder,
                                                             self.env_params.model_output_folder,
                                                             self.env_params.data_batch_size,
                                                             self.trained_rotations, self.hybrid_type)
            if self.model_params:
                tmp_model_params = dict(self.model_params)
                tmp_model_params = DictAttr(tmp_model_params)
                tmp_model_params.hybrid = False
                classic_model.change_default_params(tmp_model_params)
            classic_model.load_model(original_model)

        shape_img_buf, region_of_interest = self.deep_learning_model.get_image_buf_shape_and_roi()
        image_buf = np.zeros(shape_img_buf)
        state_buf = np.zeros(self.env_params.model_car_stage_shape)

        image_folder = os.path.join(test_folder, "images")
        preprocess = PreProcess()
        preprocess.prepare_test_data([test_folder])
        dataframe = preprocess.get_test_data()
        image_names = glob.glob(os.path.join(image_folder, "*.png"))
        for index, item in enumerate(read_images_from_path(image_names)):

            img1d = item[region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3], 0:3].astype(float)/255
            filename = image_names[index].split("\\")
            filename = filename[len(filename)-1]
            car_state = dataframe[filename]
            user_steering = car_state[0][0]

            image_buf[0] = img1d
            state_buf[0] = np.array(car_state[1])

            result = self.deep_learning_model.predict_result(image_buf, state_buf)
            predicted_steering = result[0][0]
            new_steering = round(float(predicted_steering), 2)
            #print("Predicted: " + str(predicted_steering) + " post_process:" + str(new_steering))
            #print("Should predict: " + str(user_steering))

            classic_results = None
            if classic_model:
                cl_result = classic_model.predict_result(image_buf, state_buf)
                classic_results = cl_result[0][0]
                classic_results = round(float(classic_results), 2)
                #print(classic_results)

            fig = plt.figure(figsize=(10, 3))
            plt.axis('off')
            axs = fig.add_subplot(1, 3, 1)
            axs.title.set_text("Original Image")
            plt.imshow(item)

            # hybrid steering
            axs = fig.add_subplot(1, 3, 2)
            axs.title.set_text("Hybrid Steering")
            ai_steering = [0,new_steering]
            user_steering_data = [0,user_steering]
            ys = [0, 1]
            axs.plot(user_steering_data, ys, 'gray')
            axs.plot(ai_steering, ys, 'blue')
            axs.set_xticks([-1, 1])
            axs.set_yticks([0, 1])
            axs.set_ylim([0, 1])
            axs.get_yaxis().get_major_formatter().labelOnlyBase = False
            axs.get_xaxis().get_major_formatter().labelOnlyBase = False

            # classic ai steering
            if classic_model:
                axs = fig.add_subplot(1, 3, 3)
                axs.title.set_text("Classic Steering")
                cl_ai_steering = [0, classic_results]
                user_steering_data = [0, user_steering]
                ys = [0, 1]
                axs.plot(user_steering_data, ys, 'gray')
                axs.plot(cl_ai_steering, ys, 'blue')
                axs.set_xticks([-1, 1])
                axs.set_yticks([0, 1])
                axs.set_ylim([0, 1])
                axs.get_yaxis().get_major_formatter().labelOnlyBase = False
                axs.get_xaxis().get_major_formatter().labelOnlyBase = False

            #plt.show()

            fig.savefig(os.path.join("output/plots", str(index) + "_" + filename))
            plt.close()
            #break


