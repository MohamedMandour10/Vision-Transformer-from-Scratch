import torch
import torch.nn as nn
from typing import Optional
from embeddings import PatchEmbedding
from encoder import TransformerBlock
from mlp import Head


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bias: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Vision Transformer (ViT) model implementation.
        
        Args:
            image_size (int): Input image size (assumed square)
            patch_size (int): Patch size for tokenization (assumed square)
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            embed_dim (int): Embedding dimension
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP hidden dimension ratio
            dropout (float): Dropout rate
            attention_dropout (float): Attention dropout rate
            bias (bool): Whether to use bias in linear layers
        """
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                bias=bias
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.head = Head(
            embed_dim=embed_dim,
            output_dim=num_classes,
            dropout=dropout
        )
        self.device = device

        Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device if self.device is not None else Device)
        
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            attn_mask (torch.Tensor, optional): Attention mask
            key_padding_mask (torch.Tensor, optional): Key padding mask
        
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask, key_padding_mask)
        
        # Classification
        return self.head(x)

