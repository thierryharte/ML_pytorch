from torch import nn
import torch

from ml_pytorch.utils.learning_rate_schedules import get_lr_scheduler


class DNN(nn.Module):
    def __init__(self, dim_in: int, num_classes: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.net = nn.Sequential(
            nn.Linear(dim_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)

    def export_model(self, model):
        class ONNXWrappedModel(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model

            def forward(self, x):
                logits = self.model(x)
                return torch.nn.functional.softmax(logits, dim=1)

        return ONNXWrappedModel(model)


def get_model(input_size, num_classes, device, lr, lr_schedule, n_epochs, mean, std):
    model = DNN(input_size, num_classes, mean, std).to(device)
    print(model)

    # Multi-class single-label classification:
    #   model outputs raw logits [N, C]
    #   targets are int64 class indices [N]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(lr_schedule, optimizer, n_epochs)

    return model, loss_fn, optimizer, scheduler
