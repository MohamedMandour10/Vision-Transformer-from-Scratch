import torch
import torch.nn as nn
from typing import Optional


class MLPBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.3,
        bias: bool = True,
        activation: Optional[nn.Module] = None):
        """
        MLPBlock is a module used in the Transformer architecture. 
        It consists of two fully connected linear layers with a GELU activation function in between.
        
        Args:
        - embed_dim: The input and output dimension of the block.
        - mlp_ratio: The ratio of the hidden dimension to the input/output dimension.
        - dropout: The dropout ratio.
        - bias: Whether to use bias in the linear layers.
        - activation: The activation function to use. Defaults to GELU.
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        # First linear layer
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        
        self.act = activation or nn.GELU()
        
        self.dropout1 = nn.Dropout(dropout)
        
        # Second linear layer
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights of the linear layers.
        """
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # Initialize bias to zero
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the MLPBlock module.
        
        Args:
        - x: The input tensor.
        
        Returns:
        - The output tensor after passing through the module.
        """
        # First linear layer
        x = self.fc1(x)
        
        # Activation function
        x = self.act(x)
        
        # Dropout
        x = self.dropout1(x)
        
        # Second linear layer
        x = self.fc2(x)
        
        # Dropout
        x = self.dropout2(x)
        
        return x


class Head(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        """
        Initialize the Head module.
        
        Args:
        - embed_dim: The input embedding dimension.
        - output_dim: The output dimension.
        - dropout: The dropout ratio.
        """
        super().__init__()
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            # The final linear layer
            nn.Linear(embed_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Head module.
        
        Args:
        - x: The input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
        - The output tensor of shape (batch_size, output_dim).
        """
        # The [CLS] token is the first token in the sequence, and it is used to represent the entire sequence.
        x = x[:, 0]
        # The input is normalized to have zero mean and unit variance.
        x = self.norm(x)
        # The final linear layer is used to produce the output of shape (batch_size, output_dim).
        return self.head(x)
