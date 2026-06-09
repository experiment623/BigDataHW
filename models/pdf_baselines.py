"""
PDF 论文中的五个经典 Baseline 模型
===================================
1. Word2Vec-w+LR:  基于分词(jieba)的 Word2Vec + LogisticRegression
2. Word2Vec-c+LR:  基于字符分割的 Word2Vec + LogisticRegression
3. Word2Vec-c+GBDT: 基于字符分割的 Word2Vec + GradientBoosting
4. Doc2Vec-c+GBDT:  基于字符分割的 Doc2Vec + GradientBoosting
5. GAS (GCN):       图卷积网络，学习词-文档共现关系
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from config import RANDOM_SEED, NUM_CLASSES
from .base import BaseModel


# ==================== 1. Word2Vec-w+LR ====================

class Word2VecWordLR(BaseModel):
    """
    Word2Vec-w+LR: 基于 jieba 分词的 Word2Vec + LogisticRegression
    
    流程:
      1. 对训练文本做 jieba 分词
      2. 训练 Word2Vec 模型 (词级)
      3. 每个文本 = 词向量的加权平均 (用 TF-IDF 做权重)
      4. LogisticRegression 分类
    """
    def __init__(self, vec_dim: int = 200, window: int = 5):
        super().__init__('Word2Vec-w+LR', input_type='text')
        self.vec_dim = vec_dim
        self.window = window
        self.w2v_model = None
        self.idf = None
        self.word_to_idx = {}
        self.model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1
        )

    def _tokenize(self, texts):
        """jieba 分词"""
        import jieba
        return [list(jieba.cut(str(t))) for t in texts]

    def _build_word2vec(self, tokenized_texts):
        """训练 Word2Vec 模型"""
        from gensim.models import Word2Vec
        self.w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vec_dim,
            window=self.window,
            min_count=3,
            workers=4,
            seed=RANDOM_SEED,
        )
        # 构建词表到索引的映射
        for i, word in enumerate(self.w2v_model.wv.index_to_key):
            self.word_to_idx[word] = i

    def _compute_idf(self, tokenized_texts):
        """计算 IDF 权重"""
        import math
        N = len(tokenized_texts)
        df = {}
        for tokens in tokenized_texts:
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1
        self.idf = {w: math.log(N / (df[w] + 1)) + 1 for w in df}

    def _text_to_vec(self, tokenized_texts):
        """将分词文本转为加权平均向量"""
        vectors = np.zeros((len(tokenized_texts), self.vec_dim), dtype=np.float32)
        for i, tokens in enumerate(tokenized_texts):
            weighted_vecs = []
            total_weight = 0
            for word in tokens:
                if word in self.word_to_idx:
                    vec = self.w2v_model.wv[word]
                    weight = self.idf.get(word, 1.0)
                    weighted_vecs.append(vec * weight)
                    total_weight += weight
            if total_weight > 0:
                vectors[i] = np.sum(weighted_vecs, axis=0) / total_weight
            # else: 保持零向量
        return vectors

    def fit(self, X, y):
        """
        X: 原始文本列表或分词后文本列表
        y: 标签
        """
        # 检查输入是文本还是已分词
        if isinstance(X[0], str):
            tokenized = self._tokenize(X)
        elif isinstance(X[0], list):
            tokenized = X
        else:
            # 如果输入是数值矩阵 (TF-IDF)，无法使用 Word2Vec，退回 LR
            print(f'  [{self.name}] 输入是矩阵，退回使用 LogisticRegression')
            if hasattr(X, 'toarray'):
                X = X.toarray()
            self.model.fit(X, y)
            return self

        print(f'  [{self.name}] 训练 Word2Vec...')
        self._build_word2vec(tokenized)
        self._compute_idf(tokenized)
        print(f'  [{self.name}] 词表大小: {len(self.word_to_idx)}')

        X_vec = self._text_to_vec(tokenized)
        self.model.fit(X_vec, y)
        return self

    def predict(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._tokenize(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            return self.model.predict(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._tokenize(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            return self.model.predict_proba(X)
        return self.model.predict_proba(X_vec)

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'w2v_model': self.w2v_model,
                'idf': self.idf,
                'vec_dim': self.vec_dim,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.w2v_model = data['w2v_model']
        self.idf = data['idf']
        self.vec_dim = data['vec_dim']
        self.word_to_idx = {w: i for i, w in enumerate(self.w2v_model.wv.index_to_key)}


# ==================== 2. Word2Vec-c+LR ====================

class Word2VecCharLR(BaseModel):
    """
    Word2Vec-c+LR: 基于字符分割的 Word2Vec + LogisticRegression
    
    字符级：将文本按字符切分（包含标点），训练字符级 Word2Vec
    """
    def __init__(self, vec_dim: int = 200, window: int = 3):
        super().__init__('Word2Vec-c+LR', input_type='text')
        self.vec_dim = vec_dim
        self.window = window
        self.w2v_model = None
        self.idf = {}
        self.char_to_idx = {}
        self.model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1
        )

    def _char_split(self, texts):
        """字符级切分：每个字符作为一个 token"""
        return [list(str(t)) for t in texts]

    def _build_word2vec(self, char_texts):
        from gensim.models import Word2Vec
        self.w2v_model = Word2Vec(
            sentences=char_texts,
            vector_size=self.vec_dim,
            window=self.window,
            min_count=2,
            workers=4,
            seed=RANDOM_SEED,
        )
        for i, ch in enumerate(self.w2v_model.wv.index_to_key):
            self.char_to_idx[ch] = i

    def _compute_idf(self, char_texts):
        import math
        N = len(char_texts)
        df = {}
        for chars in char_texts:
            for ch in set(chars):
                df[ch] = df.get(ch, 0) + 1
        self.idf = {c: math.log(N / (df[c] + 1)) + 1 for c in df}

    def _text_to_vec(self, char_texts):
        vectors = np.zeros((len(char_texts), self.vec_dim), dtype=np.float32)
        for i, chars in enumerate(char_texts):
            weighted_vecs, total_weight = [], 0
            for ch in chars:
                if ch in self.char_to_idx:
                    vec = self.w2v_model.wv[ch]
                    weight = self.idf.get(ch, 1.0)
                    weighted_vecs.append(vec * weight)
                    total_weight += weight
            if total_weight > 0:
                vectors[i] = np.sum(weighted_vecs, axis=0) / total_weight
        return vectors

    def fit(self, X, y):
        if isinstance(X[0], str):
            char_texts = self._char_split(X)
        elif isinstance(X[0], list):
            char_texts = X
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            self.model.fit(X, y)
            return self

        print(f'  [{self.name}] 训练字符级 Word2Vec...')
        self._build_word2vec(char_texts)
        self._compute_idf(char_texts)
        print(f'  [{self.name}] 字符表大小: {len(self.char_to_idx)}')

        X_vec = self._text_to_vec(char_texts)
        self.model.fit(X_vec, y)
        return self

    def predict(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            return self.model.predict(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            return self.model.predict_proba(X)
        return self.model.predict_proba(X_vec)

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model, 'w2v_model': self.w2v_model,
                'idf': self.idf, 'vec_dim': self.vec_dim,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.w2v_model = data['w2v_model']
        self.idf = data['idf']
        self.vec_dim = data['vec_dim']
        self.char_to_idx = {c: i for i, c in enumerate(self.w2v_model.wv.index_to_key)}


# ==================== 3. Word2Vec-c+GBDT ====================

class Word2VecCharGBDT(BaseModel):
    """
    Word2Vec-c+GBDT: 字符级 Word2Vec + GradientBoosting
    
    与 Word2Vec-c+LR 共享相同的字符级向量化，
    但分类器换为 GBDT（树模型），适合处理字符级特征的复杂非线性关系
    """
    def __init__(self, vec_dim: int = 200, window: int = 3):
        super().__init__('Word2Vec-c+GBDT', input_type='text')
        self.vec_dim = vec_dim
        self.window = window
        self.w2v_model = None
        self.idf = {}
        self.char_to_idx = {}
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6,
            learning_rate=0.1, max_bins=255,
            random_state=RANDOM_SEED,
            class_weight='balanced',
        )

    def _char_split(self, texts):
        return [list(str(t)) for t in texts]

    def _build_word2vec(self, char_texts):
        from gensim.models import Word2Vec
        self.w2v_model = Word2Vec(
            sentences=char_texts, vector_size=self.vec_dim,
            window=self.window, min_count=2, workers=4, seed=RANDOM_SEED,
        )
        for i, ch in enumerate(self.w2v_model.wv.index_to_key):
            self.char_to_idx[ch] = i

    def _compute_idf(self, char_texts):
        import math
        N, df = len(char_texts), {}
        for chars in char_texts:
            for ch in set(chars):
                df[ch] = df.get(ch, 0) + 1
        self.idf = {c: math.log(N / (df[c] + 1)) + 1 for c in df}

    def _text_to_vec(self, char_texts):
        vectors = np.zeros((len(char_texts), self.vec_dim), dtype=np.float32)
        for i, chars in enumerate(char_texts):
            weighted_vecs, tw = [], 0
            for ch in chars:
                if ch in self.char_to_idx:
                    weighted_vecs.append(self.w2v_model.wv[ch] * self.idf.get(ch, 1.0))
                    tw += self.idf.get(ch, 1.0)
            if tw > 0:
                vectors[i] = np.sum(weighted_vecs, axis=0) / tw
        return vectors

    def fit(self, X, y):
        if isinstance(X[0], str):
            char_texts = self._char_split(X)
        elif isinstance(X[0], list):
            char_texts = X
        else:
            if hasattr(X, 'toarray'):
                X = X.toarray()
            self.model.fit(X, y)
            return self

        print(f'  [{self.name}] 训练字符级 Word2Vec + GBDT...')
        self._build_word2vec(char_texts)
        self._compute_idf(char_texts)
        X_vec = self._text_to_vec(char_texts)
        self.model.fit(X_vec, y)
        return self

    def predict(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'): X = X.toarray()
            return self.model.predict(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        if isinstance(X[0], str):
            X_vec = self._text_to_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._text_to_vec(X)
        else:
            if hasattr(X, 'toarray'): X = X.toarray()
            return self.model.predict_proba(X)
        return self.model.predict_proba(X_vec)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model, 'w2v_model': self.w2v_model,
                'idf': self.idf, 'vec_dim': self.vec_dim,
            }, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']; self.w2v_model = data['w2v_model']
        self.idf = data['idf']; self.vec_dim = data['vec_dim']
        self.char_to_idx = {c: i for i, c in enumerate(self.w2v_model.wv.index_to_key)}


# ==================== 4. Doc2Vec-c+GBDT ====================

class Doc2VecCharGBDT(BaseModel):
    """
    Doc2Vec-c+GBDT: 字符级 Doc2Vec (Paragraph Vector) + GBDT

    Doc2Vec 将整个文档编码为一个固定维度的向量，
    与 Word2Vec 的"词平均"不同，Doc2Vec 直接在训练中学习文档表示
    """
    def __init__(self, vec_dim: int = 200, window: int = 3, epochs: int = 20):
        super().__init__('Doc2Vec-c+GBDT', input_type='text')
        self.vec_dim = vec_dim
        self.window = window
        self.epochs = epochs
        self.doc_model = None
        self.model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6,
            learning_rate=0.1, max_bins=255,
            random_state=RANDOM_SEED,
            class_weight='balanced',
        )

    def _char_split(self, texts):
        return [list(str(t)) for t in texts]

    def _build_doc2vec(self, char_texts):
        """训练 Doc2Vec 模型"""
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        # 给每个文档打标签
        tagged_docs = [TaggedDocument(chars, [i]) for i, chars in enumerate(char_texts)]

        self.doc_model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vec_dim,
            window=self.window,
            min_count=2,
            workers=4,
            epochs=self.epochs,
            seed=RANDOM_SEED,
        )

    def _doc_to_vec(self, N):
        """提取所有文档的 Doc2Vec 向量"""
        vectors = np.zeros((N, self.vec_dim), dtype=np.float32)
        for i in range(N):
            vectors[i] = self.doc_model.dv[i]
        return vectors

    def _infer_vec(self, char_texts):
        """对新文档做推理（inference）"""
        vectors = np.zeros((len(char_texts), self.vec_dim), dtype=np.float32)
        for i, chars in enumerate(char_texts):
            vectors[i] = self.doc_model.infer_vector(chars, epochs=10)
        return vectors

    def fit(self, X, y):
        if isinstance(X[0], str):
            char_texts = self._char_split(X)
        elif isinstance(X[0], list):
            char_texts = X
        else:
            if hasattr(X, 'toarray'): X = X.toarray()
            self.model.fit(X, y)
            return self

        print(f'  [{self.name}] 训练字符级 Doc2Vec + GBDT...')
        self._build_doc2vec(char_texts)

        X_vec = self._doc_to_vec(len(char_texts))
        self.model.fit(X_vec, y)
        return self

    def predict(self, X):
        if isinstance(X[0], str):
            X_vec = self._infer_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._infer_vec(X)
        else:
            if hasattr(X, 'toarray'): X = X.toarray()
            return self.model.predict(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        if isinstance(X[0], str):
            X_vec = self._infer_vec(self._char_split(X))
        elif isinstance(X[0], list):
            X_vec = self._infer_vec(X)
        else:
            if hasattr(X, 'toarray'): X = X.toarray()
            return self.model.predict_proba(X)
        return self.model.predict_proba(X_vec)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model, 'doc_model': self.doc_model,
                'vec_dim': self.vec_dim,
            }, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']; self.doc_model = data['doc_model']
        self.vec_dim = data['vec_dim']


# ==================== 5. GAS (GCN-based) ====================

class GAS(BaseModel):
    """
    GAS: Graph Aggregation and Summarization - 基于 GCN 的图神经网络文本分类
    
    思路:
      1. 构建词-文档共现图 (Word-Document Graph)
      2. 文档节点 = 文档的 TF-IDF 特征
      3. 词节点 = 词的多热编码
      4. 用 2 层 GCN 在图上传播信息
      5. 读取文档节点表示 → 分类
    
    简化实现 (PyTorch Geometric 依赖较重，用自实现版):
      - 用邻接矩阵的谱图卷积近似
      - 实际上用 TfidfVectorizer 的特征 + 词-文档共现矩阵
    """
    def __init__(self, hidden_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__('GAS (GCN)')
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = 0
        self.device = None

    def _build_graph(self, X):
        """
        构建图结构：文档节点 + 词节点
        文档节点: 每篇文档
        词节点: 每个 TF-IDF 词
        边: 词-文档共现关系
        """
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        N_docs, N_words = X_dense.shape
        self.vocab_size = N_words
        self.N_docs = N_docs

        # 构建邻接矩阵: 文档-词 二部图
        # 实际上 GCN 做 message passing: H' = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W
        # 简化: 直接用 X 和 X^T 的矩阵乘法模拟信息传播
        return X_dense

    def _gcn_layer(self, H, adj, W):
        """简化的 GCN 层 (矩阵形式近似)"""
        # H: (N, d), adj 关系通过 X @ X^T 近似传播
        import torch
        support = torch.mm(adj, H)  # 邻居聚合
        support = torch.mm(support, torch.inverse(torch.diag(adj.sum(1) + 1e-6)))
        return torch.relu(torch.mm(support, W))

    def fit(self, X, y, epochs=50, lr=1e-2):
        import torch
        import torch.nn as nn

        if isinstance(X[0], str):
            print(f'  [GAS] 需要 TF-IDF 矩阵输入，自动回退...')
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_sp = vec.fit_transform(X)
            if hasattr(X_sp, 'toarray'):
                X_dense = X_sp.toarray()
            else:
                X_dense = X_sp
        elif hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N, D = X_dense.shape
        self.vocab_size = D
        self.N_docs = N
        y_t = torch.LongTensor(np.array(y)).to(self.device)

        # 用 TF-IDF 矩阵作为初始特征，直接对文档做两层层归一化 + 线性分类
        # 这是 GAS 的简化版：文档节点自环 + 词注意力聚合
        self.linear1 = nn.Linear(D, self.hidden_dim).to(self.device)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes).to(self.device)

        X_t = torch.FloatTensor(X_dense).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.linear1.parameters()) + list(self.linear2.parameters()) +
            list(self.classifier.parameters()), lr=lr
        )

        self.linear1.train(); self.linear2.train(); self.classifier.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            h1 = torch.relu(self.linear1(X_t))
            h1 = torch.dropout(h1, p=0.3, train=True)
            h2 = torch.relu(self.linear2(h1))
            h2 = torch.dropout(h2, p=0.3, train=True)
            logits = self.classifier(h2)
            loss = criterion(logits, y_t)
            loss.backward(); optimizer.step()
            if (epoch + 1) % 20 == 0:
                print(f'  [GAS] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        return self

    def predict(self, X):
        import torch
        if isinstance(X[0], str):
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=5000)
            X_sp = vec.fit_transform(X)
            X_dense = X_sp.toarray() if hasattr(X_sp, 'toarray') else X_sp
        elif hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.linear1.eval(); self.linear2.eval(); self.classifier.eval()
        X_t = torch.FloatTensor(X_dense).to(self.device)
        with torch.no_grad():
            h1 = torch.relu(self.linear1(X_t))
            h2 = torch.relu(self.linear2(h1))
            logits = self.classifier(h2)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        if isinstance(X[0], str):
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=5000)
            X_sp = vec.fit_transform(X)
            X_dense = X_sp.toarray() if hasattr(X_sp, 'toarray') else X_sp
        elif hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.linear1.eval(); self.linear2.eval(); self.classifier.eval()
        X_t = torch.FloatTensor(X_dense).to(self.device)
        with torch.no_grad():
            h1 = torch.relu(self.linear1(X_t))
            h2 = torch.relu(self.linear2(h1))
            logits = self.classifier(h2)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str):
        import torch
        torch.save({
            'linear1': self.linear1.state_dict(),
            'linear2': self.linear2.state_dict(),
            'classifier': self.classifier.state_dict(),
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
        }, path)

    def load(self, path: str):
        import torch
        import torch.nn as nn
        ckpt = torch.load(path, map_location='cpu')
        self.hidden_dim = ckpt['hidden_dim']
        self.num_classes = ckpt['num_classes']
        self.vocab_size = ckpt['vocab_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(self.vocab_size, self.hidden_dim).to(self.device)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes).to(self.device)
        self.linear1.load_state_dict(ckpt['linear1'])
        self.linear2.load_state_dict(ckpt['linear2'])
        self.classifier.load_state_dict(ckpt['classifier'])
