from transformers import ViTForImageClassification

def create_vit_2d(num_labels, pretrained_model='google/vit-base-patch16-224-in21k'):
    """Create a 2D Vision Transformer model."""
    model = ViTForImageClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model