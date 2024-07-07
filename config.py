class Config:
    # Data parameters
    data_dir = "path/to/your/data"
    input_shape = (96, 96, 96)
    num_classes = 2

    # Training parameters
    batch_size = 2
    num_epochs = 100
    learning_rate = 1e-4

    # Model parameters
    vit_patch_size = (16, 16, 16)
    vit_hidden_size = 768
    vit_mlp_dim = 3072
    vit_num_layers = 12
    vit_num_heads = 12

    # 3D CNN parameters (example for a simple 3D CNN)
    cnn_filters = [64, 128, 256, 512]
    cnn_kernel_size = (3, 3, 3)

    # Device
    device = "cuda"  # or "cpu" if you don't have a GPU