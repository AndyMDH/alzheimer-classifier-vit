# models.py
import torch.nn.functional as F


class CNN3DModel(pl.LightningModule):
    def __init__(self, config):
        super(CNN3DModel, self).__init__()
        model_config = config['model']
        self.learning_rate = config['train']['learning_rate']
        self.num_classes = model_config['num_classes']

        # Define a simple 3D CNN architecture
        self.conv1 = nn.Conv3d(model_config['in_channels'], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(64 * (model_config['img_size'][0] // 4) * (model_config['img_size'][1] // 4) * (
                    model_config['img_size'][2] // 4), 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label'].long().squeeze()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label'].long().squeeze()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
