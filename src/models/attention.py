import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): The embedding dimensionality.
            num_heads (int): The number of attention heads.

        Attributes:
            num_heads (int): Stores the number of attention heads.
            head_dim (int): Stores the dimensionality of each attention head.
            scale (float): Stores the scaling factor for attention scores.
            qkv (nn.Linear): The linear layer to compute query, key and value.
            proj (nn.Linear): The linear layer to project the concatenated output.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Computes the forward pass of the MultiHeadAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
        """
        # Compute query, key and value
        b, n, c = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        # Compute scaled dot-product attention
        # q @ k^T = attention scores
        # Softmax the attention scores to get the attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Compute the output by taking a weighted sum of the value with the attention weights
        x = attn @ v
        # Concatenate the output of each attention head
        x = rearrange(x, "b h n d -> b n (h d)")
        # Project the concatenated output to the final output
        x = self.proj(x)
        return x
