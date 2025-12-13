# Deep Learning Class (VITMMA19) Project Work template


### Data Preparation

Data was of varying quality in this project, ended up using a subset of all available data. Data preparation is mainly done by a python script but pasting the data in the correct folder is a manual process.

Please download the subset from: 
- On the official OneDrive repository navigate to bullflagdetector/GFTYRV
- Download data.zip

**After downloading the data extract the zip file and place its contents inside the project's data folder so that it looks like the following inside the running container:**

*No need to copy it inside the running container you can just put it at the correct directory before starting the container, container will mount to the whole project folder because I used it for developement also*

**On the host machine inside the project folder:**

/data is the parent, and the following folders inside:
- AY1PC8
- GFTYRV
- etc...

**Inside the container it should look like:**

/work/data is the parent and inside are the following folders:
- AY1PC8
- GFTYRV
- etc...


*Although not important to know for the scripts to work each of these folders contain csv files and json files with the labelling withouth any additional folder structure*

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Máté Szendrey (GFTYRV)
- **Aiming for +1 Mark**: Maybe?

### Solution Description

Did not plan to integrate the model into a realtime solution so the task is the classification of OHLC data segments. Into one of the flag groups.

When using the inference script the user must provide a directory with csv files that each contains the ohlc data of a single flag. When running run.sh the script is currently set up to run inference on data in folder /work/inference_data. The output predictions are saved into a scv file in folder /work/src/predictions.

Performed analysis on the data and made visualizations accross several notebooks inside the notebook folder.

Looked at the problem as a classic classification task with unbalanced dataset. Chose PR curve as main metric to balance this while looking at confusion matrices and other metrics. The loss function is CrossEntropy with weighted classes to also help the imbalanced classes. I started out from a 2 layer cnn and incremented from there, added regularization to avoid overfitting to train data (there is not too much of it).

I built a one-dimensional convolutional neural network designed to classify short multivariate time-series segments. It takes sequences with eight input features and processes them through three stacked convolutional blocks that progressively increase the number of channels from 32 to 128. Each block consists of a 1D convolution with a small kernel to capture local temporal patterns, followed by batch normalization, ReLU activation, dropout for regularization, and max pooling. After the convolutional stages, the model applies adaptive average pooling to aggregate information across the entire sequence into a fixed-length representation. This representation is passed through a fully connected classifier with two hidden layers and additional dropout, ending in a six-class output layer with a softmax applied implicitly via the cross-entropy loss. Overall, the model has about 43k trainable parameters and is optimized for efficient learning of discriminative temporal patterns while controlling overfitting through normalization and dropout.

I used conda environments for package managements so alongside the requirements.txt I have an environment.yml file. There is a detailed description about how to set up the project using this.

### Extra Credit Justification

Although not necessarily aiming for +1, I implemented advanced data engineering pipeline that enabled my models to converge faster and utilize the statistical instruments which extended and transformed the original data. I also implemented hyperparameter optimization on wandb (although ended up not using the exact outputs, only observed them and drew conclusions from them). When Evaluating my models after training I used advanced metrics such as ROC-AUC curves and PR curve and also looked at the confusion matrices to fully understand model performance.

### Docker Instructions

I set up docker compose to make the project binding and starting easier for users.

*   **To run training scripts you must save the training data to the data folder in the project root as detailed in an other chapter!** [Data preparation requirements](#data-preparation).
*   logs will appear in /work/log/run.log when running the run.sh script, otherwise logs are only printed to the console inside the container
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).

Do the following steps after pasting the data inside the data folder specified in [Data preparation requirements](#data-preparation).

The first time you use this project, you need to create the Conda environment inside the Docker container.

1. **Build the image (host machine):**

    ```bash
    docker compose build
    ```

2. **Start an interactive shell in the container (host machine):**

    ```bash
    docker compose run --rm app bash
    ```

    You should now be inside the container at the `/work` directory.

3. **Initialize Conda (inside the container bash, first time only):**

    ```bash
    conda init bash
    # then restart the shell, e.g.:
    exec bash
    ```

    **Accept the Conda terms of service when prompted.**

4. **Create the environment (inside the container bash, first time only):**

    ```bash
    conda env create -f environment.yml
    # wait please :)
    ```

5. **Activate the environment (inside the container bash):**

    ```bash
    conda activate workenv
    ```

6. **Run the full pipeline via the helper script (inside the container bash):**

    ```bash
    sed -i 's/\r$//' ./run.sh # only needed the first time to replace fix line endings
    bash ./run.sh
    ```

**You will find the log file inside the log directory of the project on both your host and on the container**

For subsequent runs with the same container you only need to:

- Start a shell in the container: `docker compose run --rm app bash`
- Activate the environment: `conda activate workenv`
- Run the pipeline: `bash ./run.sh`


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
