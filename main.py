import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from GatedTabTransformer import GatedTabTransformer
import uuid
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy

# Dictionary defining datasets and their metadata.
datasets = {
    "BLASTCHAR": {
        "NAME": "blastchar.csv",
        "POSITIVE_CLASS": "Yes",
        "CONT_COUNT": 2
    },
    "INCOME": {
        "NAME": "income_evaluation.csv",
        "POSITIVE_CLASS": " >50K",
        "CONT_COUNT": 5
    },
    "BANK": {
        "NAME": "bank-full.csv",
        "POSITIVE_CLASS": "yes",
        "CONT_COUNT": 5
    }
}

def extract_ml_data(dataframe, positive_class=1, cont_count=10):
    """
    Extract categorical, continuous, and target values from a dataframe.

    This function processes a dataframe by extracting:
    - Target values (binary classification)
    - Continuous features
    - Categorical features (encoded as numeric)

    Args:
        dataframe (pandas.DataFrame): Input dataframe to process.
        positive_class (int or any, optional): Value considered as the positive class. Defaults to 1.
        cont_count (int, optional): Number of continuous columns from the end of the dataframe. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - Continuous features (numpy array): Extracted numerical columns.
            - Categorical features (numpy array): Encoded categorical columns.
            - Target values (numpy array): Binary target variable.
    """
    # Extract the target column (assumed to be the last column in the dataframe).
    targets_source = dataframe.iloc[:, -1:]
    target = np.zeros(targets_source.shape)

    # Convert the target column into binary values based on the positive class.
    for i in range(targets_source.shape[0]):
        target[i] = 1 if targets_source.iloc[i, 0] == positive_class else 0

    # Extract continuous features from the last 'cont_count' columns, excluding the target column.
    cont = dataframe.iloc[:, -(cont_count + 1):-1].to_numpy()

    # Extract categorical features from the remaining columns at the beginning of the dataframe.
    categ = dataframe.iloc[:, 0:-(cont_count + 1)].copy()

    # Encode categorical features using numerical codes for model compatibility.
    for col in categ.columns:
        categ[col] = pd.Categorical(categ[col]).codes

    return cont, categ.to_numpy(), target

def validate(model, cont_data, categ_data, target_data, device="cuda", val_batch_size=1):
    """
    Evaluate the model's performance on a validation dataset.

    This function computes predictions for the provided validation data, 
    calculates the ROC AUC score, and returns it as a metric of performance.

    Args:
        model (torch.nn.Module): The model to validate.
        cont_data (numpy array): Continuous features for validation.
        categ_data (numpy array): Categorical features for validation.
        target_data (numpy array): True target values for validation.
        device (str, optional): Device to perform computations on ("cuda" or "cpu"). Defaults to "cuda".
        val_batch_size (int, optional): Batch size for validation. Defaults to 1.

    Returns:
        float: The Area Under the Curve (AUC) score for the validation data.
    """
    model = model.eval()  # Set the model to evaluation mode.
    results = np.zeros((categ_data.shape[0], 1))  # Initialize results array.

    # Perform validation in batches.
    for i in range(categ_data.shape[0] // val_batch_size):
        x_categ = torch.tensor(categ_data[val_batch_size * i:val_batch_size * i + val_batch_size]).to(dtype=torch.int64, device=device)
        x_cont = torch.tensor(cont_data[val_batch_size * i:val_batch_size * i + val_batch_size]).to(dtype=torch.float32, device=device)

        # Make predictions and store the sigmoid-transformed outputs.
        pred = model(x_categ, x_cont)
        results[val_batch_size * i:val_batch_size * i + val_batch_size, 0] = torch.sigmoid(pred).squeeze().cpu().detach().numpy()

    # Calculate the ROC curve and AUC score.
    fpr, tpr, _ = metrics.roc_curve(target_data[:results.shape[0]], results[:, 0])
    area = metrics.auc(fpr, tpr)
    model = model.train()  # Reset the model to training mode.
    return area

if __name__ == "__main__":
    """
    Main entry point for training and evaluating a GatedTabTransformer model.

    This script performs the following steps:
    - Loads a dataset based on user selection.
    - Splits the dataset into training, validation, and test sets.
    - Preprocesses the data using `extract_ml_data`.
    - Trains a GatedTabTransformer model using the training data.
    - Validates the model on the validation data and implements early stopping.
    - Evaluates the model on the test data and prints the final AUC score.
    """
    print("Choose a dataset to train the model on:")
    for i, dataset_name in enumerate(datasets.keys(), start=1):
        print(f"{i}. {dataset_name}")
    choice = int(input("Enter the number corresponding to the dataset: ")) - 1
    dataset_name = list(datasets.keys())[choice]
    dataset_info = datasets[dataset_name]

    data_name = dataset_info["NAME"]
    positive_class = dataset_info["POSITIVE_CLASS"]
    num_cont = dataset_info["CONT_COUNT"]

    print(f"Loading the dataset '{dataset_name}'...")

    # Load the selected dataset.
    df = pd.read_csv(data_name, header=0)
    num_cat = df.iloc[:, 0:-(num_cont + 1)].nunique().to_list()

    # Split the dataset into training, validation, and test sets.
    train_length, val_length = int(df.shape[0] * 0.65), int(df.shape[0] * 0.15)
    train_df = df.iloc[:train_length, :]
    val_df = df.iloc[train_length:train_length + val_length, :]
    test_df = df.iloc[train_length + val_length:, :]

    # Preprocess the data.
    train_cont, train_categ, train_target = extract_ml_data(train_df, positive_class, num_cont)
    val_cont, val_categ, val_target = extract_ml_data(val_df, positive_class, num_cont)
    test_cont, test_categ, test_target = extract_ml_data(test_df, positive_class, num_cont)

    # Model configuration parameters.
    config = {
        "batch_size": 256,
        "patience": 5,
        "initial_lr": 1e-3,
        "scheduler_gamma": 0.1,
        "scheduler_step": 8,
        "encoder_heads": 8,
        "encoder_depth": 6,
        "encoder_dim": 8,
        "mlp_depth": 6,
        "mlp_dimension": 64,
        "dropout": 0.2,
        "output_dim": 1,
        "criterion": nn.BCEWithLogitsLoss(),
        "max_epochs": 200,
        "log_interval_steps": 10,
    }

    # Set up the model and training utilities.
    device = torch.device("cuda")
    model = GatedTabTransformer(
        category_sizes=num_cat,
        num_numerical_features=train_cont.shape[1],
        encoder_dim=config["encoder_dim"],
        output_dim=config["output_dim"],
        encoder_depth=config["encoder_depth"],
        encoder_heads=config["encoder_heads"],
        attention_dropout=config["dropout"],
        feedforward_dropout=config["dropout"],
        mlp_layers=config["mlp_depth"],
        mlp_hidden_dim=config["mlp_dimension"],
    )

    model = model.train().to(device=device)
    criterion = config["criterion"]
    optimizer = optim.AdamW(params=model.parameters(), lr=config["initial_lr"], amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    running_loss = 0.0
    max_auc = 0
    best_model_dict = None
    waiting = 0
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    log_interval = config["log_interval_steps"]
    patience = config["patience"]

    print("Training...")
    # Training loop with early stopping.
    for epoch in range(max_epochs):
        for i in range(train_categ.shape[0] // batch_size):
            optimizer.zero_grad()

            # Prepare batch data.
            x_categ = torch.tensor(train_categ[batch_size * i:batch_size * i + batch_size]).to(dtype=torch.int64, device=device)
            x_cont = torch.tensor(train_cont[batch_size * i:batch_size * i + batch_size]).to(dtype=torch.float32, device=device)
            y_target = torch.tensor(train_target[batch_size * i:batch_size * i + batch_size]).to(dtype=torch.float32, device=device)

            # Forward pass and loss computation.
            pred = model(x_categ, x_cont)
            loss = criterion(pred, y_target)
            running_loss += loss

            # Backward pass and optimization.
            loss.backward()
            optimizer.step()

            # Log training progress.
            if i % log_interval == log_interval - 1:
                print(f"Epoch: {epoch + 1}, Iteration: {i + 1}/{train_categ.shape[0] // batch_size}, Loss value: {running_loss / log_interval}")
                running_loss = 0.0

        print("Epoch finished. Evaluating...")
        running_loss = 0.0
        current_auc = validate(model, val_cont, val_categ, val_target, device=device)
        print("Validation AUC: ", current_auc)
        scheduler.step()

        # Check for improvement and early stopping.
        if current_auc > max_auc:
            max_auc = current_auc
            best_model_dict = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1

            if waiting >= patience:
                break

        print(f"Patience: {waiting}/{patience}")

    print("Training Done! 💥")
    torch.save(best_model_dict, f"./models/model_{uuid.uuid4()}")

    # Load the best model and evaluate on the test data.
    model.load_state_dict(best_model_dict)
    auc = validate(model, test_cont, test_categ, test_target, device=device)

    print("AUC value: ", auc)