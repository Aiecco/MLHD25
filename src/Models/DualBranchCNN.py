import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchCNN(nn.Module):
    def __init__(self, input_channels=1, img_size=(128, 128), gender_dim=0):
        """
        input_channels: Numero di canali dell'immagine (es. 1 per radiografie in scala di grigi)
        img_size: Dimensioni dell'immagine in input (altezza, larghezza)
        gender_dim: Dimensione del vettore che rappresenta il genere (se lo integri come feature)
        """
        super(DualBranchCNN, self).__init__()

        # --- Branch 1: Immagini Pooled ---
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # dimezza le dimensioni
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # --- Branch 2: Heatmaps ---
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calcolo dimensione feature dopo convoluzioni e pooling
        # Supponiamo che l'immagine abbia dimensione (H, W) e venga dimezzata 2 volte
        pooled_H, pooled_W = img_size[0] // 4, img_size[1] // 4
        branch_feat_dim = 32 * pooled_H * pooled_W

        # Layer fully-connected per la fusione delle due branche (e il genere, se fornito)
        total_feat_dim = branch_feat_dim * 2 + gender_dim  # somma delle due branche e, eventualmente, le feature del genere

        self.fc = nn.Sequential(
            nn.Linear(total_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Output: regressione per l'età
        )

    def forward(self, pooled_input, heatmap_input, gender_input=None):
        # Elaborazione Branch 1
        x1 = self.branch1(pooled_input)
        x1 = x1.view(x1.size(0), -1)

        # Elaborazione Branch 2
        x2 = self.branch2(heatmap_input)
        x2 = x2.view(x2.size(0), -1)

        # Concatenazione delle feature
        if gender_input is not None:
            # Se il genere è fornito, deve avere la forma [batch_size, gender_dim]
            x = torch.cat([x1, x2, gender_input], dim=1)
        else:
            x = torch.cat([x1, x2], dim=1)

        # Passaggio attraverso la rete fully-connected
        age_output = self.fc(x)
        return age_output

