import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Generate_Model:
    def __init__(self,model, dataset, num_epochs = 150):
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
    def evaluate_model(self, model, dataloader, device, criterion):
        model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for gnn_feature, target_actions in dataloader:
                # observations: [batch_size, num_agents, channels, height, width]
                batch_size, num_agents = gnn_feature.shape[0], gnn_feature.shape[1]
                # Flatten observations: [batch_size*num_agents, channels, height, width]
                gnn_feature = gnn_feature.view(batch_size * num_agents,
                                             gnn_feature.shape[2]).to(device)
                # Flatten target actions: [batch_size*num_agents]
                target_actions = target_actions.view(-1).to(device)
                # Move edge_index to device
            
                # Forward pass
                predictions = model(gnn_feature)  # [batch_size*num_agents, num_actions]
                loss = criterion(predictions, target_actions)
                total_loss += loss.item() * target_actions.size(0)
            
                # Compute accuracy
                predicted_labels = torch.argmax(predictions, dim=1)
                total_correct += (predicted_labels == target_actions).sum().item()
                total_samples += target_actions.size(0)
                
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def train_model(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for gnn_feature, target_action in self.train_loader:
                batch_size, num_agents = gnn_feature.shape[0], gnn_feature.shape[1]
                gnn_feature = gnn_feature.view(batch_size * num_agents,
                                                gnn_feature.shape[2]).to(self.device)
                target_action = target_action.view(-1).to(self.device)
                optimizer.zero_grad()
                predictions = self.model(gnn_feature)
                loss = self.criterion(predictions, target_action)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(self.train_loader)
            val_loss, val_accuracy = self.evaluate_model(self.model, self.val_loader, self.device, self.criterion)
            scheduler.step(val_loss)
            print(f"Epoch [{epoch}/{self.num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_accuracy * 100:.2f}%")
        self.save_model()
    def test_model(self):
        test_loss, test_accuracy = self.evaluate_model(self.model, self.test_loader, self.device, self.criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_loss, test_accuracy
    def save_model(self):
        torch.save(self.model.state_dict(), "trained_model_2.pth")
        print("Model saved successfully!")