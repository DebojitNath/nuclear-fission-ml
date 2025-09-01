# Nuclear Fission Simulation & ML Prediction

I simulated a nuclear chain reaction using Python, then trained a Markov-style ML model to predict neutron counts step-by-step. The plot below shows how closely the ML predictions track the physical simulation. This is how machine learning can approximate stochastic processes in physics...

## Overview
This project simulates a simplified nuclear fission chain reaction and uses **machine learning models** to predict neutron counts and classify the reactor state.  

It demonstrates:

- **Markov Chain Regression**: Predicts the number of neutrons in the next step.  
- **Classification Model**: Predicts the reactor state (`extinguished`, `equilibrium`, `supercritical`).  
- Visual comparison between simulation results and ML predictions.  


## Folder Structure

Nuclear-Fission/
│
├── data/
│   ├── simulation_results.csv   # Raw data from simulations
│   └── ml_dataset.csv           # Dataset for training ML models
│
├── models/
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   └── clf_label_encoder.pkl
│
├── src/
│   ├── main.py                  # Main file for simulation visualization
│   ├── save_results.py          # Save simulation results
│   ├── visualization.py         # Visualization functions
│   ├── comparison.py            # Compare ML models with simulation
│   ├── ml_model.py              # Train ML regression and classification models
│   ├── simulation.py            # Simulation functions
│   └── dataset_gen.py           # Generate dataset for ML training
│
└── README.md                    # Project README


## Requirements

- Python ≥ 3.10  
- Install packages:

```bash
pip install pandas matplotlib scikit-learn joblib
```


## Usage

1. Run a single simulation
python src/main.py

-Visualizes neutron count, multiplication factor (k_eff), and aggregate results.

2. Generate multiple simulation runs (for dataset)
python src/main.py

- Choose the option to generate multiple runs.
- Saves results to data/ml_dataset.csv for ML training.

3. Train ML models
python src/ml_model.py

- Trains Markov Chain Regression and Random Forest Classifier.
- Saves models in models/.

4. Compare simulation vs ML predictions 
python src/comparison.py

- Visualizes true vs predicted neutron counts.
- Shows classification results (extinguished, equilibrium,supercritical).

## Notes

- Markov Chain Concept: The model uses the previous step’s neutron count and reactor state as input to predict the next step.
- Randomness: Initial neutrons and multiplication constants are varied to generate diverse training data.
- Classification Function: Maps keff to reactor state for plotting and ML evaluation.

## License 

Please do not modify the original files in this repository.
You are welcome to create your own scripts that use these files 
and add extra features or experiments in separate code files.
