import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Mo_Star import MoStar
from Encoder import PaperCNN
from GNN_file import PaperGNN
from MLP_Action import PaperMLP
from Adjacency_Matrix import adj_mat
from Extarct_Actions import Action_Extractor


class decentralizedModel(torch.nn.Module):
    def __init__(self, cases, tensors, rfov, adjacency_matrix, num_of_robots, k_hops, num_epochs=150,
                 checkpoint_path='checkpoint.pth'):
        super().__init__()
        self.k = k_hops
        self.cases = cases
        self.rfov = rfov
        self.adjacency_matrix = adjacency_matrix
        self.num_of_robots = num_of_robots
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path

        # Data preparation

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Model components
        self.rank = nn.Embedding(num_of_robots+1, 32).to(device)
        self.cnn = PaperCNN(2).to(device)
        self.gnn = PaperGNN(input_dim=160,output_dim=160,k = 2).to(device)
        self.mlp = PaperMLP(input_dim=160,output_dim=5).to(device)

        # Combine parameters
        all_parameters = list(self.cnn.parameters())+ list(self.rank.parameters()) + list(self.gnn.parameters()) + list(self.mlp.parameters())

        # Create DataLoaders (ensure divisible by batch size * num_of_robots)
        self.dataset = self.prepare_data(tensors)
        limit = len(self.dataset) - (len(self.dataset) % (num_of_robots * 64))
        self.dataset = self.dataset[:limit]
        train_size = int(0.7 * len(self.dataset)) - (int(0.7 * len(self.dataset)) % (num_of_robots * 64))
        val_size = int(0.15 * len(self.dataset)) - (int(0.15 * len(self.dataset)) % (num_of_robots * 64))
        test_size = len(self.dataset) - train_size - val_size
        test_size = test_size - (test_size % (num_of_robots * 64))
        train_ds, val_ds, test_ds = random_split(self.dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(all_parameters, lr=1e-3, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # Checkpoint state
        self.start_epoch = 1
        self._load_checkpoint_if_exists()

    def _load_checkpoint_if_exists(self):
        """
        Load training state if checkpoint file exists.
        """
        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint '{self.checkpoint_path}'...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.rank.load_state_dict(checkpoint['rank_state_dict'])
            self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
            self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
            self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}.")
        else:
            print("No checkpoint found, starting training from scratch.")

    def save_checkpoint(self, epoch):
        """
        Save current training state to checkpoint.
        """
        state = {
            'epoch': epoch,
            'rank_state_dict' : self.rank.state_dict(),
            'cnn_state_dict': self.cnn.state_dict(),
            'gnn_state_dict': self.gnn.state_dict(),
            'mlp_state_dict': self.mlp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(state, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}.")

    def evaluate_model(self, dataloader):
        self.rank.eval()
        self.cnn.eval()
        self.gnn.eval()
        self.mlp.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for tensor, target_actions, adj_step, rank in dataloader:
                bs, na = tensor.shape[0], tensor.shape[1]
                tensor = tensor.view(bs * na, tensor.shape[2], tensor.shape[3], tensor.shape[4]).to(self.device)
                rank = rank.to(self.device)
                adj_step = adj_step.to(self.device)
                target_actions = target_actions.view(-1).to(self.device)
                rank_out = self.rank(rank).view(bs , na, -1)
                cnn_out = self.cnn(tensor).view(bs, na, -1)
                cnn_with_ranks = torch.cat((cnn_out, rank_out), dim=2)
                gnn_batch = torch.stack([self.gnn(cnn_with_ranks[i], adj_step[i]) for i in range(bs)], dim=0)
                gnn_out = gnn_batch.view(bs * na, -1)
                mlp_out = self.mlp(gnn_out)
                loss = self.criterion(mlp_out, target_actions)
                total_loss += loss.item() * target_actions.size(0)
                preds = torch.argmax(mlp_out, dim=1)
                total_correct += (preds == target_actions).sum().item()
                total_samples += target_actions.size(0)
        return total_loss / total_samples, total_correct / total_samples

    def train_model(self):
        success_rate = 0
        test_success = 0
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.rank.train()
            self.cnn.train()
            self.gnn.train()
            self.mlp.train()
            running_loss = 0
            for tensor, target_action, adj_step, rank in self.train_loader:
                bs, na = tensor.shape[0], tensor.shape[1]
                tensor = tensor.view(bs * na, tensor.shape[2], tensor.shape[3], tensor.shape[4]).to(self.device)
                adj_step = adj_step.to(self.device)
                rank = rank.to(self.device)
                target_action = target_action.view(-1).to(self.device)
                self.optimizer.zero_grad()
                rank_out = self.rank(rank).view(bs , na, -1)
                cnn_out = self.cnn(tensor).view(bs, na, -1)
                cnn_with_ranks = torch.cat((cnn_out, rank_out), dim=2)
                gnn_batch = torch.stack([self.gnn(cnn_with_ranks[i], adj_step[i]) for i in range(bs)], dim=0)
                gnn_out = gnn_batch.view(bs * na, -1)
                mlp_out = self.mlp(gnn_out)
                loss = self.criterion(mlp_out, target_action)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(self.train_loader)
            val_loss, val_acc = self.evaluate_model(self.val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch % 4 ==0:
              test_success = self.test_scenario(num_of_cases=500,random_seed = epoch+100)
              if test_success > success_rate:
                self.save_checkpoint(epoch)
                success_rate = test_success
            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Acc: {val_acc * 100:.2f}% "
                  f"LR: {current_lr:.6f} "
                  f"Success Rate: {test_success}")
            # Save checkpoint at each epoch


        print("Training complete.")
        print(f"success rate = {success_rate}")

    def test_model(self):
        test_loss, test_acc = self.evaluate_model(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")
        return test_loss, test_acc

    def prepare_data(self, tensors, cases = None):
        if cases == None:
          cases = self.cases
        dataset = []
        action_extractor = Action_Extractor(cases, self.num_of_robots)
        actions = action_extractor.extract()
        adj_obj = adj_mat(cases, self.rfov)
        adj_cases = adj_obj.get_adj_mat()
        rank_list = list(range(1,self.num_of_robots+1))
        rank_list.reverse()
        for idx in range(len(tensors)):
            # ch1 = tensors[idx]['channel 1']
            ch2 = tensors[idx]['channel 2']
            ch3 = tensors[idx]['channel 3']

            for step in range(len(ch2)):
                batch = []
                for r in range(self.num_of_robots):
                    batch.append(np.stack([ch2[step][r], ch3[step][r]]))
                data_tensor = torch.tensor(np.array(batch), dtype=torch.float32)
                act_tensor = torch.tensor(np.array(actions[idx][step]), dtype=torch.int64)
                adj_tensor = torch.tensor(np.array(adj_cases[idx][step]), dtype=torch.float32)
                ranks_tensor =torch.tensor(rank_list,dtype=torch.long)
                if ranks_tensor.shape[0] != self.num_of_robots:
                    print(ranks_tensor.shape[0])
                    print("false")
                dataset.append((data_tensor, act_tensor, adj_tensor,ranks_tensor))
        return dataset