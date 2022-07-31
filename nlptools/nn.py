from cgi import test
from pyexpat import model
import pandas as pd
import json
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline)
from datasets import Dataset, dataset_dict
from sklearn.model_selection import train_test_split




class TransformerClassifier():
    def __init__(self, num_labels=2, model_path=None, language_model="distilbert-base-uncased"):
        """Initialize a tokenizer and model if the classifier is not already finetuned"""
        if model_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(language_model, num_labels=num_labels)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            with open(f"{model_path}/label_map.json", "r") as fp:
                self.label_map = json.loads(fp.read())

    
    def train(self, train_data, test_data=None, test_size=.2, learning_rate=2e-5, batch_size=16, epochs=3, weight_decay=.01):
        """Fine-tune model on the given dataset"""
        self.label_map = {label:index for index, label in enumerate(train_data["label"].unique())}
        self.reverse_map = {value:key for key, value in self.label_map.items()}
        train_data["label"] = train_data["label"].replace(self.label_map)
        if test_data is None:
            train_data, test_data = train_test_split(train_data, test_size = test_size,
            stratify=train_data["label"])
        else:
            test_data["label"] = test_data["label"].replace(self.label_map)
        
        def transformer_tokenize(examples):
            """Tokenize, pad, and truncate the given text"""
            return self.tokenizer(examples["text"], truncation=True, padding=True)

        # prepare the datasets for the models
        train_data = Dataset.from_pandas(train_data)
        test_data = Dataset.from_pandas(test_data)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        train_data = train_data.map(transformer_tokenize, batched=True)
        test_data = test_data.map(transformer_tokenize, batched=True)

        train_data = train_data.remove_columns(["text"])
        test_data = test_data.remove_columns(["text"])
        # print(train_data)

        # prepare the trainer 
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator)

        self.trainer.train()

    def save_model(self, model_path="model_file"):
        self.trainer.save_model(model_path)
        with open(f"{model_path}/label_map.json", "w") as fp:
            json.dump(self.reverse_map, fp)

    
    def predict(self, examples):
        """Predict on new samples
        TODO: Update doc strings
        TODO: Write function to map the desired input label
        TODO: Add logic for a list or for a single example - output the relevant probabilities
        """
        pipeline = TextClassificationPipeline(model=self.model, tokenizer = self.tokenizer)
        prediction = pipeline(examples)
        return prediction


        

        

        



if __name__ == "__main__":
    train_data = pd.read_csv("test-data/train.csv")
    train_data.rename(columns={"author":"label"}, inplace=True)
    train_data = train_data[["text", "label"]]
    train_data = train_data.sample(frac=.1)
    classifier = TransformerClassifier(num_labels=3, model_path="model_file")
    # classifier.train(train_data=train_data, epochs=1)
    # classifier.save_model()
    results = classifier.predict(train_data["text"].head().tolist())
    print(results)



    
