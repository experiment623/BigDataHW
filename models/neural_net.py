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

    def _slice_to_tensor(self, X_slice):
        import torch

        if hasattr(X_slice, 'toarray'):
            X_slice = X_slice.toarray()
        elif not isinstance(X_slice, np.ndarray):
            X_slice = np.array(X_slice)

        return torch.as_tensor(X_slice, dtype=torch.float32)

    def _batch_indices(self, n_samples, batch_size, shuffle=False):
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            yield indices[start:start + batch_size]

    def fit(self, X, y, epochs=10, batch_size=64):
        import torch
        import torch.nn as nn

        self._build_model()
        y = np.asarray(y, dtype=np.int64)
        n_samples = len(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0
            for batch_idx in self._batch_indices(n_samples, batch_size, shuffle=True):
                bx = self._slice_to_tensor(X[batch_idx]).to(self.device)
                by = torch.as_tensor(y[batch_idx], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 5 == 0:
                print(f'  [{self.name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/max(n_batches, 1):.4f}')
        return self

    def predict(self, X, batch_size=256):
        import torch

        if self.model is None:
            raise RuntimeError('Model not trained')
        self.model.eval()
        preds = []
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        with torch.no_grad():
            for batch_idx in self._batch_indices(n_samples, batch_size, shuffle=False):
                bx = self._slice_to_tensor(X[batch_idx]).to(self.device)
                preds.append(self.model(bx).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

    def predict_proba(self, X, batch_size=256):
        import torch

        if self.model is None:
            raise RuntimeError('Model not trained')
        self.model.eval()
        probas = []
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        with torch.no_grad():
            for batch_idx in self._batch_indices(n_samples, batch_size, shuffle=False):
                bx = self._slice_to_tensor(X[batch_idx]).to(self.device)
                probas.append(torch.softmax(self.model(bx), dim=1).cpu().numpy())
        return np.vstack(probas) if probas else np.empty((0, self.num_classes), dtype=np.float32)

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
