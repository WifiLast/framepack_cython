
import torch
import torch.nn as nn
from typing import Dict, Any

class HiddenStateRelationshipTrainer(nn.Module):
    """
    A module to learn the relationship between the input and output hidden states of a transformer block.
    It is designed to be trained online with data from each generation pass.
    """

    def __init__(self, hidden_dim: int, lr: float = 1e-4, device: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._device = device

        # A simple MLP to model the transformation.
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        ).to(self._device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in self._device else None

    def to(self, device):
        self._device = device
        self.model.to(device)
        # Re-initialize optimizer and scaler if device changes, to move their states
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(device) else None
        return self

    @property
    def device(self):
        return self._device

    def train_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        """
        Performs a single training step on a batch of input/target hidden states.

        Args:
            input_tensor: The hidden state input to a block.
            target_tensor: The hidden state output from a block (the residual).
        """
        self.model.train()
        
        input_tensor = input_tensor.to(self.device).detach()
        target_tensor = target_tensor.to(self.device).detach()

        self.optimizer.zero_grad()

        if self.scaler:
            with torch.cuda.amp.autocast():
                prediction = self.model(input_tensor)
                loss = self.loss_fn(prediction, target_tensor)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: # CPU path
            prediction = self.model(input_tensor)
            loss = self.loss_fn(prediction, target_tensor)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the trainer, including model and optimizer.
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'hidden_dim': self.hidden_dim,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the trainer's state.
        """
        if not isinstance(state_dict, dict):
            print("Warning: Relationship trainer state_dict is invalid, skipping load.")
            return

        self.lr = state_dict.get('lr', self.lr)
        
        # Ensure model architecture matches before loading weights
        if self.hidden_dim != state_dict.get('hidden_dim'):
            print(f"Warning: Hidden dimension mismatch for relationship trainer. Expected {self.hidden_dim}, got {state_dict.get('hidden_dim')}. Re-initializing model.")
            self.hidden_dim = state_dict.get('hidden_dim', self.hidden_dim)
            self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            ).to(self.device)

        self.model.load_state_dict(state_dict['model_state_dict'])
        
        # Re-initialize optimizer and load its state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if 'optimizer_state_dict' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        self.model.to(self.device)
        print("Loaded relationship trainer state.")

