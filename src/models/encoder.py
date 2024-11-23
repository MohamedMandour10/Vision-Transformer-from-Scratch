import torch
import torch.nn as nn
from typing import Optional
from mlp import MLPBlock

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    Implements the standard transformer architecture with pre-norm design.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=bias,
            batch_first=True
        )
        
        # MLP block
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            bias=bias
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attn_mask: Optional attention mask
            key_padding_mask: Optional mask for padded tokens
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention block with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.dropout(x)
        x = residual + x
        
        # MLP block with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x