import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from src.models.architectures.patchembedding3d import PatchEmbedding3D  # Import the patch embedding module


class ViT3DM8(nn.Module):
    def __init__(self, img_size=(128, 128, 128), patch_size=8, in_chans=1, num_classes=2, embed_dim=512, depth=8,
                 num_heads=8):
        super(ViT3DM8, self).__init__()
        self.patch_embed = PatchEmbedding3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Classification token
        self.dropout = nn.Dropout(0.1)

        # Vision Transformer encoder (from timm library)
        self.transformer = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                                             num_heads=num_heads, num_classes=num_classes)

        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc_dropout = nn.Dropout(0.2)  # Dropout before the classification head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # Apply patch embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand class token for batch

        x = torch.cat((cls_tokens, x), dim=1)  # Prepend class token
        x = x + self.pos_embed  # Add position embeddings
        x = self.dropout(x)

        x = self.transformer(x)  # Apply Transformer encoder
        x = x[:, 0]  # Take the output of the class token

        x = self.fc_dropout(x)  # Apply dropout before classification
        x = self.fc(x)  # Classification head
        return x
