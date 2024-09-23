import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score
from fusions_bimodal import ConcatEarly, CrossAttention, TensorFusion, ModalityGatedFusion


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


audio_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
audio_model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960").to(device)

bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = RobertaModel.from_pretrained('roberta-base').to(device)


class MyDataset(Dataset):
    def __init__(self, text, audio, labels):
        self.text = text
        self.audio = audio
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.audio[idx], self.labels[idx]

# Data loading with your prepared data
batch_size = 64
train_dataset = MyDataset(text_train, audio_train, labels_train)
valid_dataset = MyDataset(text_valid, audio_valid, labels_valid)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 16)
        self.out = nn.Linear(16, 4)  # Adjust output size for classification task
        # self.out = nn.Linear(16, 2) # for dementia

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def train_classifier(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for audio, text, labels in data_loader:
        inputs = fusion_model(text.to(device), audio.to(device))
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio, text, labels in data_loader:
            inputs = fusion_model(text.to(device), audio.to(device))
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    return accuracy

# Generate pseudo-labels using the trained classifier
def generate_pseudo_label(model, data_loader):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for audio, text, _ in data_loader:
            inputs = fusion_model(text.to(device), audio.to(device))
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pseudo_labels.append(preds)
    
    return torch.cat(pseudo_labels)


# Compare and generate consensus pseudo-labels from acoustic, linguistic, and model predictions
def multi_view_pseudo_label(acoustic_labels, linguistic_labels, model_labels):
    return (model_labels == acoustic_labels) | (model_labels == linguistic_labels)


# Reinitialize the model weights (to avoid overfitting in iterative processes)
def reinitialize_classifier(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# Hyperparameters
I_max = 40  # Max iterations
num_epochs = 30  # Number of epochs per iteration
no_improvement_limit = 2  # Early stopping criteria
best_performance = -np.inf
no_improvement_count = 0


model = NeuralNet().to(device)
fusion_model = ModalityGatedFusion()  # Fusion model is loaded here
optimizer = optim.AdamW(
    list(model.parameters()) + list(fusion_model.parameters()), 
    lr=1e-4, eps=1e-8, weight_decay=1e-5
)
criterion = nn.CrossEntropyLoss()


# Iterative process
def iteration_process(model, val_loader, D_c, D_u, L_a, L_l, optimizer, criterion):
    global best_performance, no_improvement_count

    for iteration in range(I_max):
        print(f"Iteration {iteration + 1}/{I_max}")

        if iteration == 0:
            D_c_new = D_c  # Initialize D_c with high-confidence data

        for epoch in range(num_epochs):
            # Train classifier on the current high-confidence dataset
            train_loss = train_classifier(model, DataLoader(D_c_new, batch_size=batch_size), optimizer, criterion)
            print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

            # Evaluate on the validation set
            val_acc = evaluate_model(model, val_loader)
            print(f"Validation Accuracy: {val_acc:.4f}")

        # Early stopping
            best_performance = val_acc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= no_improvement_limit:
                print("No improvement for two consecutive iterations, stopping.")
                break

        # Generate pseudo-labels for low-confidence data (D_u)
        L_h = generate_pseudo_label(model, DataLoader(D_u, batch_size=batch_size))

        # Filter pseudo-labels based on multi-view agreement
        confident_labels = multi_view_pseudo_label(L_a, L_l, L_h)

        # Randomly remove 20% of the initial high-confidence data
        removal_count = int(0.2 * len(D_c))
        D_c_keep = random.sample(D_c, len(D_c) - removal_count)

        # Update high-confidence dataset
        D_c_new = [(d, l) for d, l, conf in zip(D_u[0], L_h, confident_labels) if conf]
        D_c_new = D_c_new + D_c_keep

        # Reinitialize the model
        reinitialize_classifier(model)
        print(f"Updated confident data size: {len(D_c_new)}")

iteration_process(model, valid_loader, D_c, D_u, L_a, L_l, optimizer, criterion)
