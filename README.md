Overview
The objective is to perform model training and testing using MLOps principles such as reproducibility, version control, and CI integration.

Assignment Structure
| Branch | Purpose |
|---------|----------|
| `main` | Final merged code |
| `dtree` | Decision Tree Regressor implementation |
| `kernelridge` | Kernel Ridge Regressor + CI workflow |


Setup & Usage

1. Clone repo and create venv
clone this Repo and create a virtual environment in the folder and install requirements.
````
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
````

2. Run Decision Tree
````
python train.py
````

4. Run Kernel Ridge
````
python train2.py
````
