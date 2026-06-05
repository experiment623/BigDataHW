"""
MLP 神经网络模型 (PyTorch)
"""
import numpy as np
from .base import BaseModel
from config import NUM_CLASSES


class SimpleMLP(BaseModel):
    """TF-IDF + 简单 MLP (PyTorch)"""
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__('TF-IDF + MLP')
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        self.model = MLP(self.input_dim, self.num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X, y, epochs=10, batch_size=64):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        self._build_model()
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_t = torch.FloatTensor(X if isinstance(X, np.ndarray) else np.array(X))
        y_t = torch.LongTensor(np.array(y))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f'  [{self.name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
        return self

    def predict(self, X):
        import torch
        if self.model is None:
            raise RuntimeError('Model not trained')
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_t = torch.FloatTensor(X if isinstance(X, np.ndarray) else np.array(X)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        if self.model is None:
            raise RuntimeError('Model not trained')
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_t = torch.FloatTensor(X if isinstance(X, np.ndarray) else np.array(X)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(X_t), dim=1).cpu().numpy()

    def save(self, path: str):
        import torch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
        }, path)

    def load(self, path: str):
        import torch
        import torch.nn as nn
        ckpt = torch.load(path, map_location='cpu')
        self.input_dim = ckpt['input_dim']
        self.num_classes = ckpt['num_classes']

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        self.model = MLP(self.input_dim, self.num_classes)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
