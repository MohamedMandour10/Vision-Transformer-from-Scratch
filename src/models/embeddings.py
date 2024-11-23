import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Initializes the PatchEmbedding module.

        Args:
            image_size (int): The size of the input image (assumed to be square). Default is 224.
            patch_size (int): The size of each patch (assumed to be square). Default is 16.
            in_channels (int): The number of input channels in the image. Default is 3 (e.g., RGB images).
            embed_dim (int): The dimension of the embedding space. Default is 768.

        Attributes:
            image_size (int): Stores the input image size.
            patch_size (int): Stores the patch size.
            num_patches (int): The total number of patches extracted from the image.
            projection (nn.Sequential): A sequential container to project image patches into an embedding space.
            cls_token (nn.Parameter): A learnable classification token added to the sequence of patches.
            pos_embedding (nn.Parameter): A learnable position embedding added to the sequence of patches and classification token.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        """
        Performs the forward pass of the PatchEmbedding module.

        Args:
            x (torch.Tensor): Input tensor representing the batch of images.
                Shape: (batch_size, num_channels, image_height, image_width)

        Returns:
            torch.Tensor: Output tensor representing the embedded patches and classification token.
                Shape: (batch_size, num_patches + 1, embed_dim)
        """
        b = x.shape[0]  # batch size
        x = self.projection(x)

        # Add classification token to each sequence
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding
        return x