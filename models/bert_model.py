"""
BERT 微调模型
"""
import numpy as np
from .base import BaseModel
from config import NUM_CLASSES


class BERTClassifier(BaseModel):
    """BERT 中文预训练模型微调"""
    def __init__(self, model_name: str = 'bert-base-chinese',
                 num_classes: int = NUM_CLASSES, max_len: int = 256,
                 batch_size: int = 16):
        super().__init__(f'BERT({model_name})', input_type='text')
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = None
        self.device = None

    def _init_model(self):
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_classes
        ).to(self.device)

    def _tokenize(self, texts):
        return self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_len, return_tensors='pt'
        )

    def fit(self, texts, labels, epochs=3, lr=2e-5):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup
        self._init_model()
        texts = [str(t) for t in texts]
        tokens = self._tokenize(texts)
        labels_t = torch.LongTensor(np.array(labels))
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_ids, attention_mask, batch_labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=batch_labels
                )
                loss = output.loss
                loss.backward(); optimizer.step(); scheduler.step()
                total_loss += loss.item()
            print(f'  [{self.name}] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
        return self

    def predict(self, texts):
        import torch
        texts = [str(t) for t in texts]
        tokens = self._tokenize(texts)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=tokens['input_ids'].to(self.device),
                attention_mask=tokens['attention_mask'].to(self.device)
            )
            return output.logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, texts):
        import torch
        texts = [str(t) for t in texts]
        tokens = self._tokenize(texts)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=tokens['input_ids'].to(self.device),
                attention_mask=tokens['attention_mask'].to(self.device)
            )
            return torch.softmax(output.logits, dim=1).cpu().numpy()

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
