from typing import List

import torch
import torch.nn as nn

import pandas as pd

from transformers import AutoTokenizer, DistilBertModel
import tqdm
import wandb
import datetime


def get_data_loader(config, filename, tokenizer, device, shuffle=False):
    df = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        names=["label", "category", "word", "position", "text"],
    )

    start_sentiment_word_delimiter = "## "
    end_sentiment_word_delimiter = " ##"

    labels = df["label"].apply(lambda x: 1 if x == "positive" else 0)
    positions = df["position"].str.split(":")
    words = df["word"].tolist()
    texts = df["text"].tolist()

    for i in range(len(texts)):
        start, end = int(positions[i][0]), int(positions[i][1])
        texts[i] = texts[i][:start] + start_sentiment_word_delimiter + words[i] + end_sentiment_word_delimiter + texts[i][end:]

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(
        device
    )

    words_encodings = tokenizer(
        words, truncation=True, padding=True, return_tensors="pt"
    ).to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encodings["input_ids"]),
        torch.tensor(encodings["attention_mask"]),
        torch.tensor(words_encodings["input_ids"]),
        torch.tensor(labels.tolist()),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle
    )

    return loader


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
    """

    ############################################# complete the classifier class below

    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.

        """

        self.config = {
            "freeze_bert": False,  # True if you want to freeze the bert layers
            "model_name": "distilbert-base-uncased",
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "n_epochs": 5,
            "batch_size": 32,
        }

        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = DistilBertModel.from_pretrained(self.model_name)

        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.bert_model.config.hidden_size, 2),
            ]
        )


        wandb.init(
            project="nlp-assignment",
            name=f"{datetime.datetime.now()}-{self.model_name}",
            config=self.config,
        )
        self.logger = wandb

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        train_loader = get_data_loader(
            self.config, train_filename, self.tokenizer, device, shuffle=True
        )

        val_loader = get_data_loader(self.config, dev_filename, self.tokenizer, device)

        self.bert_model.to(device)

        if self.config["freeze_bert"]:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.classifier.to(device)

        optimizer = torch.optim.Adam(
            list(self.bert_model.parameters()) + list(self.classifier.parameters()),
            lr=self.config["learning_rate"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["n_epochs"]*len(train_loader)
        )
        criterion = nn.CrossEntropyLoss()

        pbar = tqdm.tqdm(range(self.config["n_epochs"]))
        for epoch in range(self.config["n_epochs"]):
            self.bert_model.train()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids, attention_mask, words, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                if self.config["freeze_bert"]:
                    with torch.no_grad():
                        outputs = self.bert_model(input_ids, attention_mask)
                else:
                    outputs = self.bert_model(input_ids, attention_mask)
                for i, layer in enumerate(self.classifier):
                    if i == 0:
                        outputs = layer(outputs.last_hidden_state[:, 0, :])
                    else:
                        outputs = layer(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                self.scheduler.step()
                self.logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

                pbar.set_description(
                    f"Loss: {loss.item()} - iter: {i}/{len(train_loader)}"
                )

            self.bert_model.eval()
            correct = 0
            total = 0
            tp, fp, tn, fn = 0, 0, 0, 0
            for batch in val_loader:
                input_ids, attention_mask, words, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                with torch.no_grad():
                    outputs = self.bert_model(input_ids, attention_mask)
                for i, layer in enumerate(self.classifier):
                    if i == 0:
                        outputs = layer(outputs.last_hidden_state[:, 0, :])
                    else:
                        outputs = layer(outputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()
                # self.logger.log(
                #     {
                #         "val/loss": loss.item(),
                #     }
                # )
            f1_score = 2 * tp / (2 * tp + fp + fn)
            self.logger.log(
                {
                    "val/acc": correct / total,
                    "val/epoch": epoch,
                    "val/f1": f1_score,
                }
            )

            pbar.update(1)
        pbar.close()

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        loader = get_data_loader(self.config, data_filename, self.tokenizer, device)

        self.bert_model.to(device)
        self.classifier.to(device)

        self.bert_model.eval()

        predictions = []
        for batch in loader:
            input_ids, attention_mask, words, labels = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with torch.no_grad():
                outputs = self.bert_model(input_ids, attention_mask)
            for i, layer in enumerate(self.classifier):
                if i == 0:
                    outputs = layer(outputs.last_hidden_state[:, 0, :])
                else:
                    outputs = layer(outputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(["positive" if x == 1 else "negative" for x in predicted])

        return predictions