import argparse
from game.environment import Environment
import os
from multiprocessing import Process
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def main(model_name=None, env_params=None, train_folders=None, training=False, qml_rotations=None, hybrid_type=0, driving=False, test_folder=None, origin_model=None):
    env = Environment()
    env.add_hybrid_type(hybrid_type)

    if qml_rotations and len(qml_rotations) > 0:
        env.add_qml_rotations(qml_rotations)

    if env_params:
        #print("Using new params")
        param_data = None
        with open(env_params) as json_file:
            param_data = json.load(json_file)
        env.init_with_new_values(param_data)

    if training and train_folders:
        newDirList = []
        for x in os.scandir(train_folders):
            if x.is_dir():
                newDirList.append(x.path)

        if len(newDirList) > 0:
            env.start_training_ai(newDirList, model_name=model_name)
        else:
            env.start_training_ai([train_folders], model_name=model_name)

    elif model_name:
        env.prepare_deep_learning_model(model_name)
        if driving:
            env.start_ai_track()
        else:
            env.start_test(test_folder, origin_model)

    else:
        raise Exception("No model or training folder is defined!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Internal script endpoint for testing")
    parser.add_argument("--model_name", dest="model_name", action="store",
                        help="This parameter defines the model name which should be used for the ai sim")
    parser.add_argument("--env_params", dest="env_params", action="store",
                        help="This parameter defines a json path which contains parameters for the environment,"
                             "pre processing, training and driving step")
    parser.add_argument("--train_folder", dest="train_folder", action="store",
                        help="This parameter defines the training data and is also an indication for the "
                             "environment to create a new model")

    args = parser.parse_args()
    training = False
    driving = False
    test_folder = "output/test_data"
    origin_model = "normal/top_model.46-0.0013866-0.0001746.h5"

    if not training:
        model_config = None
        with open("model_settings.json") as json_file:
            model_config = json.load(json_file)


        tmp_model_name = "model.('100',)-0.0011060.h5"
        index = len(model_config)-3
        if tmp_model_name:
            for item_index, item in enumerate(model_config):
                if tmp_model_name in item["model"]:
                    index = item_index
                    break
        print(model_config[index]["model"] + " type: " + str(model_config[index]["hybrid_type"]))
        model_name = model_config[index]["model"]
        #trained_rotations = [0.241, 0.248, 0.293, 0.463, -0.262, 0.315, 0.26, 0.318, 0.241, 0.248, 0.293, 0.463,
        #                     -0.262, 0.315, 0.26, 0.318]
        trained_rotations = model_config[index]["trained_rotations"]
        hybrid_type = model_config[index]["hybrid_type"]
        #print(model_name)
        main(model_name, args.env_params, args.train_folder, training, trained_rotations, hybrid_type, driving,
             test_folder, origin_model)


    else:
        hybrid_types = [1]
        process_list = []
        pretrained_model = args.model_name
        #model_name = None

        trained_rotations = None

        # 4 models are trained at the same time
        n_threads = 6
        for index in range(0, len(hybrid_types), n_threads):
            for i in range(n_threads):
                current_index = index + i
                if current_index >= len(hybrid_types):
                    break
                hybrid_type = hybrid_types[index + i]
                p = Process(target=main, args=(pretrained_model, args.env_params, args.train_folder, training, trained_rotations, hybrid_type))
                process_list.append(p)

            for item in process_list:
                item.start()

            for item in process_list:
                item.join()

            process_list.clear()


