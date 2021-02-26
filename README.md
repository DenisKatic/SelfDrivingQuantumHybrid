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

#Setup
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

# Tests
Test the first quantum hybrid model that is defined in the trained_quantum_circuits.json file with some sample images 

Note: The trained_quantum_circuits.json file contains the model path!
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

# Model training


