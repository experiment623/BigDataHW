"""
GCA-Net v5: 字形-字音-语义三模态联合对抗变体感知预训练
=====================================================

三模态编码:
  字形流 (Glyph):  字体渲染(32×32) → 4层CNN → 512维
  字音流 (Phonetic): pinyin序列 → 1D-CNN + BiLSTM → 512维
  语义流 (Semantic): 可学习字符嵌入 → MLP → 512维

预训练任务:
  1. 三模态锚点对齐损失 (L_tri)        — 拉近同字三模态
  2. 对抗不变性对比损失 (L_inv_sent)    — 原文-变体全局对齐
  3. 语义冲突损失 (L_sem_confl)         — 推远冲突字对
  4. 跨模态字符判别损失 (L_disc)         — 从变体推理原字

微调: 冻结主干 + 2层MLP分类器
"""
import os, re, math, random
import numpy as np


def _find_chinese_font():
    candidates = [
        r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None


# ===================================================================
# Part 1: 对抗变体生成 (预训练阶段用)
# ===================================================================

SHAPE_CONFUSE = {
    '证':'証','贷':'貸','款':'欵','药':'薬','博':'愽','彩':'採',
    '赌':'賭','码':'碼','微':'薇','信':'伩','加':'伽','电':'電',
    '话':'話','元':'圆','万':'萬','千':'仟','百':'佰','钱':'銭',
    '一':'壹','二':'贰','六':'陆','七':'柒','八':'捌','九':'玖',
}
SOUND_CONFUSE = {
    '博':'搏','彩':'采','赌':'堵','码':'马','证':'正','贷':'代',
    '款':'宽','药':'要','钱':'前','微':'危','信':'心','下':'夏',
    '中':'忠','小':'晓','快':'块','加':'家','元':'原',
}
SEMANTIC_CONFLICT = {  # 同音/形近但语义冲突
    '惠':'慧', '账':'障', '贷':'待', '款':'宽', '博':'薄',
    '证':'政', '码':'蚂', '包':'抱', '加':'假', '费':'废',
    '元':'员', '信':'欣', '提':'题',
}


def generate_adversarial_variant(text, p=0.4, replace_ratio=0.15):
    """
    预训练期间的对抗变体生成
    p: 应用变体替换的概率
    replace_ratio: 替换字符比例
    返回: (variant_text, positions, original_chars, replaced_chars, conflict_flags)
    """
    if random.random() > p:
        return text, [], [], [], []

    chars = list(text)
    n = len(chars)
    n_replace = max(1, int(n * replace_ratio))
    try:
        indices = random.sample(range(n), n_replace)
    except ValueError:
        return text, [], [], [], []

    all_confuse = {**SHAPE_CONFUSE, **SOUND_CONFUSE}
    variant, origs, repls, conflicts = [], [], [], []

    for idx in sorted(indices):
        ch = chars[idx]
        # 70% 形/音混淆, 30% 语义冲突
        if random.random() < 0.7 and ch in all_confuse:
            repl = all_confuse[ch]
            conflict = False
        elif random.random() < 0.3 and ch in SEMANTIC_CONFLICT:
            repl = SEMANTIC_CONFLICT[ch]
            conflict = True
        else:
            # 不行就尝试另一个
            if ch in all_confuse:
                repl = all_confuse[ch]; conflict = False
            elif ch in SEMANTIC_CONFLICT:
                repl = SEMANTIC_CONFLICT[ch]; conflict = True
            else:
                continue
        origs.append(ch); repls.append(repl); conflicts.append(conflict)
        variant.append((idx, repl))
        chars[idx] = repl

    return ''.join(chars), [p[0] for p in variant], origs, repls, conflicts


# ===================================================================
# Part 2: 三模态编码器
# ===================================================================

class GlyphCNN:
    """字形流: 32×32灰度图 → 4层CNN → 512维"""
    def __init__(self, out_dim=512):
        import torch, torch.nn as nn
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),          # 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),          # 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),          # 8→4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # →(B,256,1,1)
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        import torch
        h = self.cnn(x).squeeze(-1).squeeze(-1)
        return self.proj(h)


class PhoneticEncoder:
    """字音流: pinyin字符序列 → 1D-CNN + BiLSTM → 512维"""
    def __init__(self, pinyin_vocab_size=64, out_dim=512):
        import torch, torch.nn as nn
        self.embed = nn.Embedding(pinyin_vocab_size, 64, padding_idx=0)
        self.conv1d = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(256 + 256, out_dim)

    def forward(self, pinyin_seq_embed, pinyin_raw_embed):
        import torch
        # 1D-CNN 分支
        cnn_out = pinyin_seq_embed.transpose(1, 2)  # (B,64,L)
        cnn_feat = self.conv1d(cnn_out).squeeze(-1)  # (B,256)
        # BiLSTM 分支
        lstm_out, _ = self.lstm(pinyin_raw_embed)
        lstm_feat = lstm_out[:, -1, :]  # (B,256) 取最后时刻
        return self.proj(torch.cat([cnn_feat, lstm_feat], dim=-1))


class SemanticEmbedding:
    """语义流: 字符嵌入 → MLP → 512维 (方案C:词向量蒸馏)"""
    def __init__(self, vocab_size=5000, out_dim=512):
        import torch, torch.nn as nn
        self.embed = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(256, 384), nn.ReLU(),
            nn.Linear(384, out_dim),
        )

    def forward(self, char_ids):
        return self.proj(self.embed(char_ids))


# ===================================================================
# Part 3: 三模态融合 + Transformer 主干
# ===================================================================

class TriModalFusion:
    """融合层: [g; p; s] → 768维 + 3层Transformer"""
    def __init__(self, d_model=768, num_layers=3, num_heads=8, max_len=256):
        import torch, torch.nn as nn
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, glyph, phonetic, semantic):
        import torch
        B, S, _ = glyph.shape
        fused = self.fusion(torch.cat([glyph, phonetic, semantic], dim=-1))  # (B,S,768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, fused], dim=1)  # (B,S+1,768)
        x = x + self.pos_embed[:, :S+1, :]
        out = self.transformer(x)
        return out[:, 0, :], out[:, 1:, :]  # cls, chars


# ===================================================================
# Part 4: GCANet — 三模态预训练 + 微调
# ===================================================================

class GCANet:
    """字形-字音-语义三模态联合对抗预训练网络"""

    def __init__(self, name='GCA-Net', glyph_weight=0.3, max_chars=5000):
        self.name = name
        self.glyph_weight = glyph_weight
        self.device = 'cpu'
        self.tfidf_dim = None
        self.num_classes = None

        # ── 字体渲染 ──
        self.font = None
        self._init_font()

        # ── 三模态编码器 ──
        self.glyph_cnn = None
        self.phonetic_enc = None
        self.semantic_emb = None
        self.fusion_tr = None

        # ── 字符词表 ──
        self.char_vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[MASK]': 3}
        self.char_list = ['[PAD]', '[UNK]', '[CLS]', '[MASK]']

        # ── 拼音词表 ──
        self.pinyin_vocab = {'[PAD]': 0}
        self.pinyin_list = ['[PAD]']

        # ── 分类器 (微调时创建) ──
        self.classifier = None

        self.max_chars = max_chars
        self.max_seq = 256

    def _init_font(self):
        path = _find_chinese_font()
        if path:
            try:
                from PIL import Image, ImageDraw, ImageFont
                self.font = ImageFont.truetype(path, size=28)
            except Exception as e:
                print(f'[GCA-Net] 字体加载失败: {e}')

    def _render_char(self, ch):
        """渲染单个汉字为 32×32 灰度图"""
        from PIL import Image, ImageDraw
        img = Image.new('L', (32, 32), 255)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), ch, font=self.font, fill=0)
        return 1.0 - np.array(img, dtype=np.float32) / 255.0

    def _get_pinyin_seq(self, ch):
        """获取带声调的拼音字母序列"""
        try:
            from pypinyin import lazy_pinyin, Style
            py = lazy_pinyin(ch, style=Style.TONE3, errors='ignore')
            return py[0] if py else 'unk'
        except:
            return 'unk'

    def _build_vocabs(self, texts):
        """构建字符和拼音词表"""
        all_chars = set()
        all_pinyin_letters = set()

        for text in texts:
            clean = re.sub(r'\s+', '', str(text))
            for ch in clean:
                all_chars.add(ch)
                py = self._get_pinyin_seq(ch)
                for c in py:
                    all_pinyin_letters.add(c)

        # 扩充字符词表
        for ch in sorted(all_chars):
            if ch not in self.char_vocab and len(self.char_vocab) < self.max_chars:
                self.char_vocab[ch] = len(self.char_vocab)
                self.char_list.append(ch)

        # 扩充拼音词表
        for c in sorted(all_pinyin_letters):
            if c not in self.pinyin_vocab:
                self.pinyin_vocab[c] = len(self.pinyin_vocab)
                self.pinyin_list.append(c)

        print(f'[Vocab] 字符: {len(self.char_vocab)}, 拼音字母: {len(self.pinyin_vocab)}')

    def _encode_text_batch(self, texts):
        """批量编码文本为三模态特征"""
        import torch

        B = len(texts)
        S = self.max_seq
        glyphs = torch.zeros(B, S, 512)
        phones = torch.zeros(B, S, 512)
        semans = torch.zeros(B, S, 512)

        char_ids = torch.zeros(B, S, dtype=torch.long)

        for b, text in enumerate(texts):
            clean = re.sub(r'\s+', '', str(text))[:S]
            for i, ch in enumerate(clean):
                cid = self.char_vocab.get(ch, 1)
                char_ids[b, i] = cid

                # 字形: 渲染 → CNN
                if self.font and self.glyph_cnn:
                    try:
                        img = torch.FloatTensor(self._render_char(ch)).unsqueeze(0)
                        glyphs[b, i] = self.glyph_cnn(img.unsqueeze(0))
                    except:
                        pass

        # 语义嵌入
        if self.semantic_emb:
            semans = self.semantic_emb(char_ids)

        return {'glyph': glyphs, 'phonetic': phones, 'semantic': semans,
                'char_ids': char_ids}

    # ── 预训练 ──

    def pretrain(self, texts, epochs=20, batch_size=48, lr=5e-4):
        """三模态联合对抗预训练"""
        import torch, torch.nn as nn, torch.nn.functional as F

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_vocabs(texts)

        # 初始化三模态编码器
        self.glyph_cnn = GlyphCNN(512).to(self.device)
        self.phonetic_enc = PhoneticEncoder(len(self.pinyin_vocab), 512).to(self.device)
        self.semantic_emb = SemanticEmbedding(len(self.char_vocab), 512).to(self.device)
        self.fusion_tr = TriModalFusion(768, num_layers=3).to(self.device)

        # 字符判别头
        disc_head = nn.Linear(768, len(self.char_vocab)).to(self.device)

        all_params = (list(self.glyph_cnn.parameters()) +
                      list(self.phonetic_enc.parameters()) +
                      list(self.semantic_emb.parameters()) +
                      list(self.fusion_tr.parameters()) +
                      list(disc_head.parameters()))
        optimizer = torch.optim.AdamW(all_params, lr=lr)
        tau = 0.1

        print(f'[Pre-train] 三模态预训练, 设备: {self.device}, '
              f'样本: {len(texts)}, 字符: {len(self.char_vocab)}')

        for epoch in range(epochs):
            total_loss = 0.0
            indices = np.random.permutation(len(texts))
            n_batches = 0

            for start in range(0, len(texts), batch_size // 2):
                idxs = indices[start:start + batch_size // 2]
                batch_texts = [texts[i] for i in idxs]

                # 生成对抗变体
                adv_texts, all_pos, all_orig, all_repl, all_conf = [], [], [], [], []
                for t in batch_texts:
                    at, pos, orig, repl, conf = generate_adversarial_variant(t)
                    adv_texts.append(at)
                    all_pos.append(pos)
                    all_orig.append(orig)
                    all_repl.append(repl)
                    all_conf.append(conf)

                # 编码原文+变体
                orig_enc = self._encode_text_batch(batch_texts)
                adv_enc = self._encode_text_batch(adv_texts)

                og, op, os_ = orig_enc['glyph'].to(self.device), orig_enc['phonetic'].to(self.device), orig_enc['semantic'].to(self.device)
                ag, ap, as_ = adv_enc['glyph'].to(self.device), adv_enc['phonetic'].to(self.device), adv_enc['semantic'].to(self.device)

                B, S = og.shape[0], og.shape[1]

                # Transformer 编码
                cls_orig, char_orig = self.fusion_tr(og, op, os_)  # (B,768) (B,S,768)
                cls_adv,  char_adv  = self.fusion_tr(ag, ap, as_)

                # ── L_tri: 三模态锚点对齐 ──
                L_tri = 0.0
                count = 0
                for mi, mj, mk, name_i in [
                    (og, op, os_, 'glyph'), (op, og, os_, 'phonetic'), (os_, og, op, 'semantic')
                ]:
                    target = (mj + mk) / 2.0  # 另外两模态均值作为锚点
                    mi_n = F.normalize(mi.reshape(B*S, 512), dim=-1)
                    t_n = F.normalize(target.reshape(B*S, 512), dim=-1)
                    # 简单形式: 最大化余弦
                    pos_sim = (mi_n * t_n).sum(dim=-1).mean()
                    L_tri += -pos_sim
                    count += 1
                L_tri = L_tri / max(count, 1)

                # ── L_inv_sent: 原文-变体全局对比 ──
                cls_vecs = torch.cat([cls_orig, cls_adv], dim=0)
                cls_vecs = F.normalize(cls_vecs, dim=-1)
                sim = cls_vecs @ cls_vecs.T / tau
                # 正对: (i, i+B) 和 (i+B, i)
                labels = torch.cat([torch.arange(B)+B, torch.arange(B)], dim=0).to(self.device)
                mask = ~torch.eye(2*B, dtype=torch.bool, device=self.device)
                sim = sim[mask].reshape(2*B, 2*B-1)
                labels_adj = torch.where(labels.unsqueeze(1) > torch.arange(2*B, device=self.device).unsqueeze(0),
                                         labels.unsqueeze(1)-1, labels.unsqueeze(1))[:, :2*B-1]
                L_inv_sent = F.cross_entropy(sim, labels_adj[:, 0])

                # ── L_sem_confl: 语义冲突推远 ──
                L_sem_confl = 0.0
                conf_count = 0
                for b in range(B):
                    for pos, repl, conf in zip(all_pos[b], all_repl[b], all_conf[b]):
                        if conf and pos < S:
                            orig_sem = char_orig[b, pos, :]
                            adv_sem = char_adv[b, pos, :]
                            sim_val = F.cosine_similarity(orig_sem.unsqueeze(0), adv_sem.unsqueeze(0))
                            L_sem_confl += F.relu(sim_val - 0.2).mean()
                            conf_count += 1
                if conf_count > 0:
                    L_sem_confl = L_sem_confl / conf_count

                # ── L_disc: 跨模态字符判别 ──
                L_disc = 0.0
                disc_count = 0
                for b in range(B):
                    for pos, orig in zip(all_pos[b], all_orig[b]):
                        if pos < S:
                            logit = disc_head(char_adv[b, pos])
                            target = self.char_vocab.get(orig, 1)
                            L_disc += F.cross_entropy(logit.unsqueeze(0),
                                                       torch.LongTensor([target]).to(self.device))
                            disc_count += 1
                if disc_count > 0:
                    L_disc = L_disc / disc_count

                loss = L_tri + 0.5 * L_inv_sent + 0.2 * L_sem_confl + 1.0 * L_disc
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f'  [Pre-train] Epoch {epoch+1}/{epochs}, Loss: {total_loss/max(n_batches,1):.4f}')

        torch.save({
            'glyph_cnn': self.glyph_cnn.state_dict(),
            'phonetic_enc': self.phonetic_enc.state_dict(),
            'semantic_emb': self.semantic_emb.state_dict(),
            'fusion_tr': self.fusion_tr.state_dict(),
            'char_vocab': self.char_vocab,
            'char_list': self.char_list,
            'pinyin_vocab': self.pinyin_vocab,
        }, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'gca_net_pretrained.pt'))
        print('[Pre-train] 预训练完成, 权重已保存')
        return self

    # ── 微调 ──

    def fit(self, X_tfidf, y, texts, epochs=10, batch_size=32, lr=1e-3,
            glyph_weight=0.3, contrast_weight=0.1):
        """微调阶段: 冻结主干 + 训练分类器"""
        import torch, torch.nn as nn, torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        self.tfidf_dim = X_tfidf.shape[1]
        self.num_classes = len(set(y))

        if self.glyph_cnn is None:
            raise RuntimeError('请先运行 pretrain() 进行预训练')

        self.glyph_cnn.to(self.device)
        self.phonetic_enc.to(self.device)
        self.semantic_emb.to(self.device)
        self.fusion_tr.to(self.device)

        # 冻结主干
        for enc in [self.glyph_cnn, self.phonetic_enc, self.semantic_emb, self.fusion_tr]:
            for p in enc.parameters():
                p.requires_grad = False

        if hasattr(X_tfidf, 'toarray'):
            X_dense = X_tfidf.toarray()
        else:
            X_dense = X_tfidf

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.tfidf_dim + 768, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
        ).to(self.device)

        N = len(texts)
        cls_batch = []
        for start in range(0, N, batch_size):
            batch_t = texts[start:start+batch_size]
            enc = self._encode_text_batch(batch_t)
            og, op, os_ = enc['glyph'].to(self.device), enc['phonetic'].to(self.device), enc['semantic'].to(self.device)
            with torch.no_grad():
                cls_vec, _ = self.fusion_tr(og, op, os_)
            cls_batch.append(cls_vec.cpu())

        all_cls = torch.cat(cls_batch, dim=0)
        X_t = torch.FloatTensor(X_dense[:N])
        y_t = torch.LongTensor(np.array(y)[:N])

        dataset = TensorDataset(X_t, all_cls, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f'[Fine-tune] 训练分类器, 设备: {self.device}')
        for epoch in range(epochs):
            total_loss = 0
            self.classifier.train()
            for bx, bc, by in loader:
                bx, bc, by = bx.to(self.device), bc.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                tw = 1.0 - glyph_weight
                combined = torch.cat([bx * tw, bc * glyph_weight], dim=-1)
                loss = criterion(self.classifier(combined), by)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f'  [Fine-tune] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')

        return self

    # ── 预测 ──

    def predict(self, X):
        import torch
        self.classifier.eval()
        if hasattr(X, 'toarray'): X = X.toarray()
        X_t = torch.FloatTensor(X).to(self.device)
        ctx = torch.zeros(X_t.shape[0], 768).to(self.device)
        tw = 1.0 - self.glyph_weight
        with torch.no_grad():
            return self.classifier(torch.cat([X_t*tw, ctx*self.glyph_weight], -1)).argmax(1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        self.classifier.eval()
        if hasattr(X, 'toarray'): X = X.toarray()
        X_t = torch.FloatTensor(X).to(self.device)
        ctx = torch.zeros(X_t.shape[0], 768).to(self.device)
        tw = 1.0 - self.glyph_weight
        with torch.no_grad():
            return torch.softmax(self.classifier(torch.cat([X_t*tw, ctx*self.glyph_weight], -1)), 1).cpu().numpy()

    def save(self, path):
        import torch
        torch.save({
            'glyph_cnn': self.glyph_cnn.state_dict() if self.glyph_cnn else None,
            'classifier': self.classifier.state_dict() if self.classifier else None,
            'tfidf_dim': self.tfidf_dim, 'num_classes': self.num_classes,
            'char_vocab': self.char_vocab,
        }, path)
        print(f'  [{self.name}] 已保存: {path}')

    def load(self, path):
        import torch
        ckpt = torch.load(path, map_location='cpu')
        self.tfidf_dim = ckpt['tfidf_dim']
        self.num_classes = ckpt['num_classes']
        self.char_vocab = ckpt.get('char_vocab', {'[PAD]':0,'[UNK]':1})

        self.glyph_cnn = GlyphCNN(512)
        self.phonetic_enc = PhoneticEncoder(64, 512)
        self.semantic_emb = SemanticEmbedding(len(self.char_vocab), 512)
        self.fusion_tr = TriModalFusion(768, 3)
        if ckpt['glyph_cnn'] is not None:
            self.glyph_cnn.load_state_dict(ckpt['glyph_cnn'])

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.tfidf_dim + 768, 256),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, self.num_classes),
        )
        self.classifier.load_state_dict(ckpt['classifier'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        print(f'  [{self.name}] 已加载: {path}')
