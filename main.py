import argparse
from game.environment import Environment
import os
from multiprocessing import Process
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def main(model_name=None, env_params=None, train_folders=None, training=False, qml_rotations=None, hybrid_type=0,
         driving=False, test_folder=None, origin_model=None, output_plots_folder=None):
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
            env.start_test(test_folder, origin_model, output_plots_folder)

    else:
        raise Exception("No model or training folder is defined!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Internal script endpoint for testing")
    parser.add_argument("--env_params", dest="env_params", action="store",
                        help="This parameter defines a json path which contains parameters for the environment,"
                             "pre processing, training and driving step")
    parser.add_argument("--training_folder", dest="training_folder", action="store",
                        help="This parameter defines the training data and is also an indication for the "
                             "environment to create a new model")
    parser.add_argument("--training_flag", dest="training_flag", action="store",
                        help="This parameter defines if a new model should be trained")
    parser.add_argument("--driving_flag", dest="driving_flag", action="store",
                        help="This parameter defines if the model should be used to run on the simulation")
    parser.add_argument("--original_model", dest="original_model", action="store",
                        help="This parameter defines the classical (pretrained) model for test comparison")
    parser.add_argument("--test_data_folder", dest="test_data_folder", action="store",
                        help="This parameter defines the location of the test data")
    parser.add_argument("--plot_output_folder", dest="plot_output_folder", action="store",
                        help="This parameter defines the output location of the comparison/test results")
    parser.add_argument("--trained_quantum_circuits", dest="trained_quantum_circuits", action="store",
                        help="This parameter defines the input json path, where the trained quantum circuit parameters are defined")
    parser.add_argument("--trained_quantum_circuits_index", dest="trained_quantum_circuits_index", action="store",
                        help="This parameter defines the specific model at the trained json file")
    parser.add_argument("--hybrid_types", dest="hybrid_types", action="store",
                        help="This parameter defines the amount and quantum hybrid types for training e.q. [1,3] -> type 1 and type 2 quantum hybrid should be trained")
    parser.add_argument("--n_threads", dest="n_threads", action="store",
                        help="This parameter defines how many threads should be used for training")


    args = parser.parse_args()
    training_flag = False
    if args.training_flag:
        training_flag = True
    driving_flag = False
    if args.driving_flag:
        driving_flag=True

    if not training_flag:
        model_config = None
        with open(args.trained_quantum_circuits) as json_file:
            model_config = json.load(json_file)

        index = int(args.trained_quantum_circuits_index)
        print(model_config[index]["model"] + " type: " + str(model_config[index]["hybrid_type"]))
        model_name = model_config[index]["model"]
        trained_rotations = model_config[index]["trained_rotations"]
        hybrid_type = model_config[index]["hybrid_type"]
        main(model_name, args.env_params, args.training_folder, training_flag, trained_rotations, hybrid_type, driving_flag,
             args.test_data_folder, args.original_model, args.plot_output_folder)


    else:
        hybrid_types = [1]
        process_list = []
        pretrained_model = args.original_model

        trained_rotations = None

        # 4 models are trained at the same time
        n_threads = args.n_threads
        for index in range(0, len(hybrid_types), n_threads):
            for i in range(n_threads):
                current_index = index + i
                if current_index >= len(hybrid_types):
                    break
                hybrid_type = hybrid_types[index + i]
                p = Process(target=main, args=(pretrained_model, args.env_params, args.train_folder, training_flag, trained_rotations, hybrid_type))
                process_list.append(p)

            for item in process_list:
                item.start()

            for item in process_list:
                item.join()

            process_list.clear()


