import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.enabled = False
import torch.nn as nn
import string
import sys

# sys.path.append('spell_crt/')
# from spell_crt.predictor import Predictor
# from dataset.add_noise import SynthesizeData
# predictor = Predictor(weight_path='spell_crt/weights/seq2seq.pth', have_att = True)
# synther = SynthesizeData()


from transformers import AutoModel, AutoTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

from transformers import pipeline





from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction", device = 'cuda')



train = pd.read_csv('vlsp_sentiment_train.csv', delimiter='\t')


test = pd.read_csv('vlsp_sentiment_test.csv', delimiter='\t')


new_df = train[['Data', 'Class']]


new_df_test = test[['Data', 'Class']]


import pandas as pd
from underthesea import word_tokenize


import regex as re
import string
import json

emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

def clean_text(text):
    text = text.lower()
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(r'([a-z]+?)\1+',r'\1', text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
    # text = re.sub(r"(\d)([^\d.])", r"\1 \2", text)
    # text = re.sub(r"([^\d.])(\d)", r"\1 \2", text)
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+",r"\1", text)
    text = text.strip()
    while text.endswith(tuple(string.punctuation+string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation+string.whitespace)):
        text = text[1:]
    text = re.sub(r"\s+", " ", text)
    return text






def preprocess_vietnamese_text(text):
    # Tokenization using underthesea
    text = clean_text(text)
    text = text.lower()
    #text = corrector(text,max_length=128)[0]['generated_text']
    
    #text = remove_stopwords(text)
    #text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # try:
    #      text = corrector(text, max_length=256)[0]['generated_text']
    # except: 
    #      pass
        
    #text = text.lower()  
    tokens = word_tokenize(text)
    
    # Join tokens back into a single string
    tokens = [token for token in tokens]

    preprocessed_text = " ".join(tokens)
    

    return preprocessed_text

# Apply text preprocessing to 'Data' column directly
new_df['Data'] = new_df['Data'].apply(preprocess_vietnamese_text)

new_df_test['Data'] = new_df_test['Data'].apply(preprocess_vietnamese_text)

# Display the updated DataFrame
print(new_df.head())


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-05
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


# from vncorenlp import VnCoreNLP

    # class PipelineConfig:
    #     rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=9000) 


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Data
        self.targets = self.data.Class
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        #segmented_text = PipelineConfig.rdrsegmenter.tokenize(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation = True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
    
    
train_size = 1
train_data=new_df.sample(frac=train_size,random_state=200)
test_data=new_df_test.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

training_set = SentimentData(train_data, tokenizer, MAX_LEN)
testing_set = SentimentData(test_data, tokenizer, MAX_LEN)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


import torch
from transformers import AutoModel

class RobertaClassWithLSTM(torch.nn.Module):
    def __init__(self):
        super(RobertaClassWithLSTM, self).__init__()
        self.l1 = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.pre_classifier = torch.nn.Linear(768,768)
        self.dropout = torch.nn.Dropout(0.1)
        # self.lstm = torch.nn.LSTM(input_size=768, hidden_size=256, num_layers=2, batch_first=True, bidirectional = True)  # Define the LSTM layer
        
        self.lstm = nn.LSTM(input_size=768, 
                        hidden_size=1024, 
                        batch_first=True, bidirectional=True, num_layers=1)
        self.classifier = torch.nn.Linear(2048, 3)  # Output size from LSTM to classifier

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)

        # LSTM input should have a shape (batch_size, sequence_length, input_size)
        lstm_input = pooler.unsqueeze(1)  # Add a dimension for sequence length
        lstm_output, _ = self.lstm(lstm_input)  # LSTM layer
        lstm_output = lstm_output[:, -1, :]  # Taking the output of the last time step
        output = self.classifier(lstm_output)
        return output
    
# class RobertaClass(torch.nn.Module):
#     def __init__(self):
#         super(RobertaClass, self).__init__()
#         self.l1 = AutoModel.from_pretrained("vinai/phobert-base")
#         self.pre_classifier = torch.nn.Linear(768, 768)
#         self.dropout = torch.nn.Dropout(0.1)
#         self.classifier = torch.nn.Linear(768, 3)

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         hidden_state = output_1[0]
#         pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = torch.nn.ReLU()(pooler)
#         pooler = self.dropout(pooler)
#         output = self.classifier(pooler)
#         return output


    
model = RobertaClassWithLSTM()
model = model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)



def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def calculate_f1_score(y_true, y_pred, num_classes):
    #print(y_true, y_pred)
    f1_scores = []
    for cls in range(num_classes):
        tp = sum((p == cls) and (t == cls) for p, t in zip(y_pred, y_true))
        fp = sum((p == cls) and (t != cls) for p, t in zip(y_pred, y_true))
        fn = sum((p != cls) and (t == cls) for p, t in zip(y_pred, y_true))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    micro_f1 = sum(f1_scores) / num_classes if num_classes > 0 else 0
    #print(micro_f1)
    return micro_f1


# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        # if _%5000==0:
        #     loss_step = tr_loss/nb_tr_steps
        #     accu_step = (n_correct*100)/nb_tr_examples 
        #     print(f"Training Loss per 5000 steps: {loss_step}")
        #     print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 


EPOCHS = 6
for epoch in range(EPOCHS):
    train(epoch)
    
def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0; n_f1 = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
            
            n_f1 += calculate_f1_score(targets, big_idx, num_classes = 3)
            

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    epoch_f1 = (n_f1 * 100)/len(testing_loader)
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    print(f"Validation F1-Score Epoch: {epoch_f1}")
    
    return epoch_accu


acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)


output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')