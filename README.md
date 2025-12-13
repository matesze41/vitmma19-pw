# Deep Learning Class (VITMMA19) Project Work template


### Data Preparation

Data was of varying quality in this project, ended up using a subset of all available data.

Please download the subset from: 
- On the official OneDrive repository navigate to bullflagdetector/GFTYRV
- Download data.zip

**After downloading the data extract the zip file and place its contents inside the project's data folder so that it looks like the following inside the running container:**

/work/data is the parent and inside is the following:
- /work/data/AY1PC8
- /work/data/GFTYRV
- etc...

*Although not important to know for the scripts to work each of these folders contain csv files and json files with the labelling withouth any additional folder structure*

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Máté Szendrey (GFTYRV)
- **Aiming for +1 Mark**: No

### Solution Description

Did not plan to integrate the model into a realtime solution so the task is the classification of OHLC data segments. Into one of the flag groups.

When using the inference script the user must provide a directory with csv files that each contains the ohlc data of a single flag. When running run.sh the script is currently set up to run inference on data in folder /work/inference_data. The output predictions are saved into a scv file in folder /work/src/predictions.

Looked at the problem as a classic classification task with unbalanced dataset. Chose PR curve as main metric to balance this while looking at confusion matrices and other metrics. The loss function is CrossEntropy with weighted classes to also help the imbalanced classes.

Performed analysis on the data and made visualizations accross several notebooks inside the notebook folder.

I used conda environments for package managements so alongside the requirements.txt I have an environment.yml file. There is a detailed description about how to set up the project using this.

### Extra Credit Justification

Although not aiming for +1, I implemented advanced data engineering pipeline that enabled my models to converge faster and utilize the statistical instruments which extended and transformed the original data. I also implemented hyperparameter optimization on wandb (although ended up not using the exact outputs, only observed them and drew conclusions from them). When Evaluating my models after training I used advanced metrics such as ROC-AUC curves and PR curve and also looked at the confusion matrices to fully understand model performance.

### Docker Instructions

I set up docker compose to make the project binding and starting easier for users.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker compose build
```

#### Run

(Inside the container it should be at /work/data).

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker compose up
```

*   **To run training scripts you must save the training data to the data folder in the project root as detailed in an other chapter!** [Data preparation requirements](#data-preparation).
*   logs will appear in /work/log/run.log when running the run.sh script, otherwise logs are only printed to the console
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


To run the sh script:



### File Structure and Functions

The repository is structured as follows:

- **`src/`** – Source code for the end‑to‑end pipeline.
    - `01-data-preprocessing.py` – Loads labeled segments from JSON/CSV, performs feature engineering, builds train/test splits, and writes processed datasets to `data/export`.
    - `train_model.py` – Uses `config.py` to train the `FlagPatternClassifier` with PyTorch Lightning, saves checkpoints and `eval_metadata.pt`, and writes training curves to `src/training_plots/`.
    - `03-evaluation.py` – Loads the best checkpoint and test data, evaluates the CNN and the baseline detector, logs metrics, and saves confusion-matrix plots to `src/evaluation_plots/`.
    - `04-inference.py` – Loads the trained model and runs inference on all CSV files in a given directory; saves predictions to `src/predictions/predictions.csv`.
    - `baseline_model.py` – Rule‑based / classical baseline methods plus helpers used for comparison in evaluation.
    - `config.py` – Central configuration for paths (e.g. `data/export`) and training hyperparameters (epochs, batch size, learning rate, etc.).
    - `utils.py` – Shared utilities, including `setup_logger` used by all scripts for consistent logging to stdout (captured in `log/run.log` when using `run.sh`).

- **`notebook/`** – Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb` – Initial exploratory data analysis and visualizations.
    - `02-label-analysis.ipynb` – Analysis of label distributions and annotation quality.
    - `03-data-preproc.ipynb`, `05-exp.ipynb`, `06-optimize.ipynb` – Prototyping of preprocessing, experiments, and hyperparameter tuning.

- **`data/`** – Input data used by the pipeline.
    - Contains subfolders such as `AY1PC8/`, `GFTYRV/`, etc., each with raw OHLC CSV files and the corresponding Label Studio JSON annotations.
    - The `data/export/` subfolder is created by `01-data-preprocessing.py` and holds derived CSV/HDF5 files and metadata for training, evaluation, and inference.

- **`log/`** – Runtime logs.
    - `run.log` – Combined log output when running the full pipeline via `run.sh`.

- **Root directory** – Project entry points and environment setup.
    - `run.sh` – Convenience script that runs the full pipeline: preprocessing → training → evaluation → inference.
    - `Dockerfile`, `compose.yml` – Docker setup for reproducible environments.
    - `requirements.txt`, `environment.yml` – Python dependency specifications.
    - `README.md` – This documentation file.
