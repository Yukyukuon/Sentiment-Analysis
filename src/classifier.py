from typing import List
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForSequenceClassification, AdamW


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
        model_name = "bert-base-uncased"
        num_labels = 3
        batch_size = 16
        self.batch_size = batch_size
        self.epochs = 10
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.best_val_accuracy = 0
        self.best_model = None


    def prepare_data(self, filename: str):
        column_names = ['Polarity', 'AspectCategory', 'Term', 'Offsets', 'Sentence']
        data = pd.read_csv(filename, delimiter='\t', header=None, names=column_names)
        dataset = Dataset.from_pandas(data)
        
        return dataset

    def preprocess_function(self, examples):
        start_offsets, end_offsets = zip(*[(int(offset.split(':')[0]), int(offset.split(':')[1]))
                                           for offset in examples['Offsets']])
        term_contexts = [sentence[start:end] for sentence, start, end in zip(examples['Sentence'], start_offsets, end_offsets)]
        encoded_inputs = self.tokenizer(examples['Sentence'], term_contexts, truncation=True, padding='max_length', max_length=50)
        polarity_to_id = {'positive': 2, 'neutral': 1, 'negative': 0}
        encoded_inputs['labels'] = [polarity_to_id[p] for p in examples['Polarity']]

        return encoded_inputs

    def convert_to_tensors(self, dataset):
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return dataset
       
    def train(self, train_filename: str, dev_filename: str, device: torch.device):

        # preprocess data 
        train_data = self.prepare_data(train_filename)
        dev_data = self.prepare_data(dev_filename)
        
        train_dataset = train_data.map(self.preprocess_function, batched=True)
        dev_dataset = dev_data.map(self.preprocess_function, batched=True)
        
        train_dataset = self.convert_to_tensors(train_dataset)
        dev_dataset = self.convert_to_tensors(dev_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size)

        self.model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-6)
        self.criterion = nn.CrossEntropyLoss()

        # train model
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
    
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
    
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
    
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                train_acc += (predictions == labels).sum().item()
    
            train_avg_loss = train_loss / len(train_loader)
            train_avg_accuracy = train_acc / len(train_loader.dataset)
            # print(f'Training Loss: {train_avg_loss:.4f}, Training Accuracy: {train_avg_accuracy:.4f}')
            
            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()
    
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    val_acc += (predictions == labels).sum().item()
                    
            val_avg_loss = val_loss / len(dev_loader)
            val_avg_accuracy = val_acc / len(dev_loader.dataset)
            # print(f'Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_avg_accuracy:.4f}\n')

            if val_avg_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_avg_accuracy
                self.best_model = copy.deepcopy(self.model.state_dict())
              
    
    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        # prepare data
        print()
        predict_data = self.prepare_data(data_filename)
        predict_dataset = predict_data.map(self.preprocess_function, batched=True)
        predict_dataset = self.convert_to_tensors(predict_dataset)
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size)
               
        if self.best_model:
            self.model.to(device)
            self.model.load_state_dict(self.best_model)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in predict_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.tolist())

        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        string_predictions = [label_map[pred] for pred in predictions]

        return string_predictions
