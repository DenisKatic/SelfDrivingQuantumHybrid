# SelfDrivingQuantumHybrid
Hybrid quantum-classical neural networks for self-driving cars

#Project description
This project aims to test and evaluate the current capabilities of variational quantum circuits based
on a hybrid ML approach, and a simplified and simulated version of a self-driving use case.

The idea is to train a classic ML model that contains multiple CNN and dense layers to predict the 
car's steering angle based on images. After a good model has been successfully trained, some of the 
weights are transferred (transfer learning) to a new model architecture, where a dense layer has 
been exchanged for a variational quantum circuit. Only the weights and parameters that have not 
been transferred are trained. Furthermore, various quantum circuits are trained and evaluated.


This project is based on [Unreal Engine](https://www.unrealengine.com/en-US/), [AirSim](https://microsoft.github.io/AirSim/), 
Nvidia's [end-to-end learning for self-driving cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
paper, and the experience in developing the Siemens/Austrian Institute of Technology AI demo, in which the entire machine learning 
workflow, including explainability, is presented in a simplified manner.

# Setup
Install project requirements
```
pip install -r requirements.txt
```
Note: One file of the installed libraries had to be modified due to some problems with keras. I also added some 
printing instructions to get the trained weights and model structure as I could not save the model information. 
```
File path: ./venv/lib/python3.7/site-packages/pennylane/qnn/keras.py

Function at line 228 added:
    def get_config(self):
        print(self.qnode.draw())
        print(self.weights)
        cfg = super().get_config()
        return cfg
```


# Model training
The env_settings.json file contains parameters which are important for the training.
```
{
    "env_params": {
        "ai_car_sim_ip": "127.0.0.3",           #imporant for the simulation
        "ai_car_sim_port": 41451,
        "model_training_folder": "output/models",
        "model_processed_folder": "images/preprocessed/", # trainings data will be preprocessed and packed into h5 files
        "model_output_folder": "output/tmp"     # folder where the models are stored
    },
    "model_params": {                           # first part is important for the classic ml model
        "seed_number": 1,
        "batch_size": 32,
        "train_zero_drop_percentage": 0,
        "val_zero_drop_percentage": 0,
        "roi": [76, 135, 0, 255],
        "activation_function": "relu",
        "padding": "same",
        "data_generator_horizontal_flip": false,
        "data_generator_brighten_range": 0,
        "learning_rate": 0.0001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "loss_function": "mse",
        "monitored_quantity": "val_loss",
        "plateau_factor": 0.5,
        "plateau_patience": 10,
        "plateau_min_learning_rate": 0.0000001,
        "plateau_verbose": 1,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0,
        "model_training_epoche": 100,
        "trainable": false,                     # this flag defines whether the weights of the classic ml should be changed  
        "hybrid": true,                         # defines if a hybrid model should be trained or not
        "qml_trainable": true                   # defines if the parameters of the hybrid model should be changed or not
    }
}
```

## For qauntum hybrid:
The env_settings.json file must be modified:
```
    "trainable": false,     # depending on whether you want to use the transfer learning or not 
    "hybrid": true,
    "qml_trainable": true
```
Command
```
python3.7 main.py --env_params env_settings.json --original_model output/models/normal/top_model.46-0.0013866-0.0001746.h5 
--hybrid_types 1,2 --n_threads 1 --training_flag true --training_folder training_data_folder
```

For classic model

The env_settings.json file must be modified:
```
    "trainable": true,
    "hybrid": false
```
Command
```
python3.7 main.py--env_params env_settings.json --training_flag true --training_folder train_folder
```


# Tests
Test the first quantum hybrid model that is defined in the trained_quantum_circuits.json file with some sample images 

Note: The trained_quantum_circuits.json file contains the model path!
Note: The env_settings.json file must be modified for testing. 
The qml_trainable parameters also defines, if the trainings weights of the trained_quantum_circuits.json file should be used or not.
```
"trainable": false,                     # this flag defines whether the weights of the classic ml should be changed  
"hybrid": true,                         # defines if a hybrid model should be trained or not
"qml_trainable": true                  # defines if the parameters of the hybrid model should be changed or not
```

Command:
```
python3.7 main.py --env_params env_settings.json --test_data_folder test --trained_quantum_circuits 
trained_quantum_circuits.json --trained_quantum_circuits_index 1
```
The use of the original_model parameter includes the classic ML in the test 
```
python3.7 main.py --env_params env_settings.json --test_data_folder test --trained_quantum_circuits 
trained_quantum_circuits.json --trained_quantum_circuits_index 1 --original_model 
output/models/normal/top_model.46-0.0013866-0.0001746.h5 
```
By adding the parameter plot_output_folder, the plots are saved in this folder and are no longer displayed 
```
python3.7 main.py --env_params env_settings.json --test_data_folder test --trained_quantum_circuits 
trained_quantum_circuits.json --trained_quantum_circuits_index 1 --original_model 
output/models/normal/top_model.46-0.0013866-0.0001746.h5 --plot_output_folder output/plots
```
