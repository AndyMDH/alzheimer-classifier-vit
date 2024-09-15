import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding for 3D Vision Transformer.
    This splits a 3D image into patches, flattens the patches, and applies a linear projection.
    """
    def __init__(self, img_size=(128, 128, 128), patch_size=16, in_chans=1, embed_dim=768):
        super(PatchEmbedding3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.patch_dim = in_chans * patch_size ** 3  # Volume of the patch
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)  # Dropout after projection

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            "Input size must match the given image size."

        # Split into non-overlapping 3D patches and flatten each patch
        x = x.unfold(2, self.patch_size, self.patch_size)  # Depth
        x = x.unfold(3, self.patch_size, self.patch_size)  # Height
        x = x.unfold(4, self.patch_size, self.patch_size)  # Width

        x = x.contiguous().view(B, C, -1, self.patch_size ** 3)  # Reshape into patches
        x = x.permute(0, 2, 1, 3)  # Permute to (batch_size, num_patches, in_chans, patch_volume)
        x = x.flatten(2)  # Flatten the patch volume into a vector
        x = self.proj(x)  # Linear projection into embed_dim
        x = self.dropout(x)  # Apply dropout
        return x
