pip install transformers
pip install torch
pip install scikit-learn

import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers import BertConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers import EarlyStoppingCallback 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('Aspect_Based.csv', sep=';')

# Tokonisasi dan format data
def process_data(dataframe, tokenizer, max_len=256):
    input_ids = []
    attention_masks = []
    aspect_ids = []
    sentiment_labels = [] 

    aspect_map = {
        'Fasilitas': 0,
        'Harga': 1,
        'Hotel': 2,
        'Kamar': 3,
        'Kolam Renang': 4,
        'Layanan': 5,
        'Lokasi': 6,
        'Sarapan': 7,
        'Staf': 8
    }

    sentiment_map = { # Renaming to sentiment_map for consistency within this function
        0: 0,
        1: 1 
    }


    for index, row in dataframe.iterrows():
        text = str(row['Review']) 
        aspect_list = str(row['Aspect']).split(',')
        sentiment = int(row['Sentiment'])

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        if aspect_list and aspect_list[0].strip() in aspect_map:
             aspect_ids.append(aspect_map[aspect_list[0].strip()])
        else:
             aspect_ids.append(-1) 

      sentiment_labels.append(sentiment)


    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_mask': torch.cat(attention_masks, dim=0),
        'aspect_ids': torch.tensor(aspect_ids), 
        'sentiment_labels': torch.tensor(sentiment_labels) 
    }

if 'df_train' in locals():
    processed_data = process_data(df_train, tokenizer)

    print("Processed Data (first sample):")
    print(f"  Input IDs: {processed_data['input_ids'][0]}")
    print(f"  Attention Mask: {processed_data['attention_mask'][0]}")
    print(f"  Aspect ID: {processed_data['aspect_ids'][0]}")
    print(f"  Sentiment Label: {processed_data['sentiment_labels'][0]}") 
    print(f"Number of samples processed: {len(processed_data['input_ids'])}")
else:
    print("DataFrame df_train not found. Please load your data first.")

# Load data training
try:
    df_train = pd.read_csv('Aspect_Based.csv', sep= ";")
    print("Data training berhasil dimuat")
except FileNotFoundError:
    print("Error: File 'Aspect_Based.csv' tidak ditemukan.")
    df_train = None # Set df_train to None if file not found


tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")

unique_aspects = set()
if df_train is not None:
    for aspect_list in df_train['Aspect'].dropna():
        for aspect in aspect_list.split(','):
            unique_aspects.add(aspect.strip())

aspect_map = {aspect: i for i, aspect in enumerate(sorted(list(unique_aspects)))}
reverse_aspect_map = {i: aspect for aspect, i in aspect_map.items()}

print("Aspect Map:")
print(aspect_map)

unique_sentiments = sorted(df_train['Sentiment'].unique().tolist())
print(f"\nUnique Sentiment values in data: {unique_sentiments}")

sentiment_map_int_to_label = {
    0: 'Negative',
    1: 'Positive'
} 
print("Sentiment Map (Integer to Label:")
print(sentiment_map_int_to_label)

# Mendefinisikan a function to process the DataFrame
def process_absa_data(dataframe, tokenizer, aspect_map, sentiment_map_int_to_label, max_len=256):
    input_ids = []
    attention_masks = []
    aspect_ids = [] # Store aspect IDs
    sentiment_labels = [] # Store sentiment labels (integers)

    for index, row in dataframe.iterrows():
        text = str(row['Review']) # Ensure text is string
        aspect_list = str(row['Aspect']).split(',') # Split aspects (handle potential NaN as string)
        sentiment = int(row['Sentiment']) # Ensure sentiment is integer

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        if aspect_list and aspect_list[0].strip() in aspect_map:
             aspect_ids.append(aspect_map[aspect_list[0].strip()])
        else:
             aspect_ids.append(-1) # Use -1 for unknown or missing aspects

        sentiment_labels.append(sentiment)

    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_mask': torch.cat(attention_masks, dim=0),
        'aspect_ids': torch.tensor(aspect_ids),
        'sentiment_labels': torch.tensor(sentiment_labels)
    }

if df_train is not None:
    processed_train_data = process_absa_data(df_train, tokenizer, aspect_map, sentiment_map_int_to_label)

    # Print the processed data structure (first sample)
    print("\nProcessed Training Data (first sample):")
    print(f"  Input IDs: {processed_train_data['input_ids'][0]}")
    print(f"  Attention Mask: {processed_train_data['attention_mask'][0]}")
    print(f"  Aspect ID: {processed_train_data['aspect_ids'][0]}")
    print(f"  Sentiment Label: {processed_train_data['sentiment_labels'][0]}")
    print(f"  Number of training samples: {len(processed_train_data['input_ids'])}")
else:
    print("\nDataFrame df_train is not loaded. Cannot proceed with data processing.")

# Mengkonversikan processed data dictionary to a TensorDataset
if df_train is not None:
    dataset = TensorDataset(processed_train_data['input_ids'],
                            processed_train_data['attention_mask'],
                            processed_train_data['aspect_ids'],
                            processed_train_data['sentiment_labels'])

    # Menghitung jumlah sampel untuk setiap split data
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size # Ensure all samples are included

    # Split data
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"\nDataset split into:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
else:
    print("\nDataset not created because df_train was not loaded.")

class IndoBERTABSA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_aspect_labels = len(aspect_map)
        self.num_sentiment_labels = len(sentiment_map_int_to_label)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.aspect_classifier = nn.Linear(config.hidden_size, self.num_aspect_labels)
        self.sentiment_classifier = nn.Linear(config.hidden_size, self.num_sentiment_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, aspect_labels=None, sentiment_labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1] # Use pooled output for classification

        pooled_output = self.dropout(pooled_output)

        aspect_logits = self.aspect_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)

        outputs = (aspect_logits, sentiment_logits,) + outputs[2:]

        if aspect_labels is not None and sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            valid_aspect_indices = (aspect_labels != -1)
            valid_sentiment_indices = (sentiment_labels != -1)

            aspect_loss = loss_fct(aspect_logits[valid_aspect_indices], aspect_labels[valid_aspect_indices])
            sentiment_loss = loss_fct(sentiment_logits[valid_sentiment_indices], sentiment_labels[valid_sentiment_indices])

            total_loss = aspect_loss + sentiment_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), aspect_logits, sentiment_logits, (hidden_states), (attentions)

# Mengimpor model pre-trained
config = BertConfig.from_pretrained("indobenchmark/indobert-base-p2")

absa_model = IndoBERTABSA(config)

print("Custom IndoBERT ABSA model instantiated.")
print(f"Number of aspect labels in model: {absa_model.num_aspect_labels}")
print(f"Number of sentiment labels in model: {absa_model.num_sentiment_labels}")




# Define a function to compute metrics (moved here to be used in TrainingArguments)
def compute_metrics(eval_pred):
    # eval_pred is a tuple: (predictions, labels)
    # In our case, predictions will be (aspect_logits, sentiment_logits)
    # and labels will be (aspect_labels, sentiment_labels)
    aspect_logits, sentiment_logits = eval_pred.predictions
    aspect_labels, sentiment_labels = eval_pred.label_ids

    # Get the predicted class indices by taking the argmax
    aspect_predictions = np.argmax(aspect_logits, axis=1)
    sentiment_predictions = np.argmax(sentiment_logits, axis=1)


    # --- Compute metrics for Aspect Prediction ---
    # Filter out invalid aspect labels (-1) if they exist
    valid_aspect_indices = (aspect_labels != -1)
    valid_aspect_labels = aspect_labels[valid_aspect_indices]
    valid_aspect_predictions = aspect_predictions[valid_aspect_indices]


    aspect_accuracy = accuracy_score(valid_aspect_labels, valid_aspect_predictions)
    # Use zero_division=0 to handle cases where there are no predictions for a class
    aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(
        valid_aspect_labels,
        valid_aspect_predictions,
        average='weighted', # Use weighted average for imbalanced classes
        zero_division=0
    )

    # --- Compute metrics for Sentiment Prediction ---
    # Filter out invalid sentiment labels (-1) if they exist
    valid_sentiment_indices = (sentiment_labels != -1)
    valid_sentiment_labels = sentiment_labels[valid_sentiment_indices]
    valid_sentiment_predictions = sentiment_predictions[valid_sentiment_indices]


    sentiment_accuracy = accuracy_score(valid_sentiment_labels, valid_sentiment_predictions)
    # Use zero_division=0 to handle cases where there are no predictions for a class
    sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(
        valid_sentiment_labels,
        valid_sentiment_predictions,
        average='weighted', # Use weighted average for imbalanced classes
        zero_division=0
    )

    return {
        "aspect_accuracy": aspect_accuracy,
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": aspect_f1,
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_precision": sentiment_precision,
        "sentiment_recall": sentiment_recall,
        "sentiment_f1": sentiment_f1,
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=20,             
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.001,              
    logging_dir='./logs',          
    logging_steps=10,            
    eval_strategy="epoch",    
    save_strategy="epoch",           
    load_best_model_at_end=True,    
    report_to="none",
    # compute_metrics=compute_metrics 
)

# Buat custom Trainer
class ABSATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): 
        inputs = {k: v.to(model.device) for k, v in inputs.items()} 

        aspect_labels = inputs.pop("aspect_labels", None)
        sentiment_labels = inputs.pop("sentiment_labels", None)

        outputs = model(**inputs, aspect_labels=aspect_labels, sentiment_labels=sentiment_labels)

        loss = outputs[0] if isinstance(outputs, tuple) else outputs

        return (loss, outputs) if return_outputs else loss

def simple_data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    aspect_labels = torch.stack([f[2] for f in features])
    sentiment_labels = torch.stack([f[3] for f in features])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'aspect_labels': aspect_labels, 
        'sentiment_labels': sentiment_labels,
    }

# Initialize the Trainer
trainer = ABSATrainer(
    model=absa_model,                    
    args=training_args,                 
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,           
    data_collator=simple_data_collator,  
    compute_metrics=compute_metrics,     
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

# Start training
print("Memulai model training...")
trainer.train()
print("Training selesai.")


# Mendefinisikan fungsi ke compute metrics
def compute_metrics(eval_pred):
    aspect_predictions, sentiment_predictions = eval_pred.predictions
    aspect_labels, sentiment_labels = eval_pred.label_ids

    aspect_predictions = np.argmax(aspect_predictions, axis=1)
    sentiment_predictions = np.argmax(sentiment_predictions, axis=1)

    valid_aspect_indices = (aspect_labels != -1)
    valid_aspect_labels = aspect_labels[valid_aspect_indices]
    valid_aspect_predictions = aspect_predictions[valid_aspect_indices]

    aspect_accuracy = accuracy_score(valid_aspect_labels, valid_aspect_predictions)
    aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(
        valid_aspect_labels,
        valid_aspect_predictions,
        average='weighted', # Use weighted average for imbalanced classes
        zero_division=0
    )

    valid_sentiment_indices = (sentiment_labels != -1)
    valid_sentiment_labels = sentiment_labels[valid_sentiment_indices]
    valid_sentiment_predictions = sentiment_predictions[valid_sentiment_indices]


    sentiment_accuracy = accuracy_score(valid_sentiment_labels, valid_sentiment_predictions)
    sentiment_precision, sentiment_recall, sentiment_f1, _ = precision_recall_fscore_support(
        valid_sentiment_labels,
        valid_sentiment_predictions,
        average='weighted', # Use weighted average for imbalanced classes
        zero_division=0
    )

    return {
        "aspect_accuracy": aspect_accuracy,
        "aspect_precision": aspect_precision,
        "aspect_recall": aspect_recall,
        "aspect_f1": aspect_f1,
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_precision": sentiment_precision,
        "sentiment_recall": sentiment_recall,
        "sentiment_f1": sentiment_f1,
    }

# Evaluasi model pada test dataset
print("Evaluating model on the test dataset...")
eval_results = trainer.evaluate(eval_dataset=test_dataset) # compute_metrics is already passed to Trainer

# Print hasil evaluasi
print("\nHasil Evaluasi:")
print(eval_results)

model_save_path = 'absa_indobert_model.pth'

# Menyimpan Model
torch.save(absa_model.state_dict(), model_save_path)

print(f"Model sukses disimpan {model_save_path}")
