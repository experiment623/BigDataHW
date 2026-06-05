"""
GCA-Net: Glyph-Contrastive Alignment Network
============================================
创新方法：将中文字符分解为部件(偏旁部首) + 结构 + 拼音 + 笔画四维特征，
构建字形感知嵌入，用对比学习使形近/音近字在向量空间中靠近，
从根本上增强对同形字替换、同音字替换等对抗攻击的鲁棒性。

论文思路：
  - 传统方法(BERT/TF-IDF)将'博'和'搏'视为完全不同的token
  - GCA-Net 通过字形分解发现它们共享部首'十'和相似结构
  - 对比学习主动拉近形近字、拉开无关字，形成字形感知的嵌入空间
"""
import os
import numpy as np
import pickle
import re
import jieba
from collections import defaultdict

# ==================== 第一部分：中文字符拆解数据库 ====================

# ---- 汉字 → 拼音映射（常用字，去声调）----
PINYIN_MAP = {
    '博': 'bo', '搏': 'bo', '薄': 'bo', '膊': 'bo',
    '彩': 'cai', '采': 'cai', '菜': 'cai', '踩': 'cai',
    '赌': 'du', '堵': 'du', '睹': 'du', '都': 'du',
    '码': 'ma', '马': 'ma', '妈': 'ma', '骂': 'ma',
    '证': 'zheng', '正': 'zheng', '政': 'zheng', '整': 'zheng',
    '办': 'ban', '半': 'ban', '伴': 'ban', '扮': 'ban',
    '卡': 'ka', '咖': 'ka', '咯': 'ka',
    '贷': 'dai', '代': 'dai', '带': 'dai', '袋': 'dai',
    '款': 'kuan', '宽': 'kuan',
    '药': 'yao', '要': 'yao', '腰': 'yao', '摇': 'yao',
    '钱': 'qian', '前': 'qian', '千': 'qian', '签': 'qian',
    '信': 'xin', '心': 'xin', '新': 'xin', '辛': 'xin',
    '微': 'wei', '危': 'wei', '位': 'wei', '味': 'wei',
    '元': 'yuan', '原': 'yuan', '圆': 'yuan', '员': 'yuan',
    '万': 'wan', '完': 'wan', '玩': 'wan', '晚': 'wan',
    '一': 'yi', '以': 'yi', '已': 'yi', '易': 'yi',
    '大': 'da', '达': 'da', '答': 'da', '打': 'da',
    '加': 'jia', '家': 'jia', '佳': 'jia', '价': 'jia',
    '号': 'hao', '好': 'hao', '毫': 'hao', '浩': 'hao',
    '来': 'lai', '莱': 'lai', '赖': 'lai',
    '下': 'xia', '夏': 'xia', '吓': 'xia', '虾': 'xia',
    '中': 'zhong', '忠': 'zhong', '钟': 'zhong', '终': 'zhong',
    '上': 'shang', '尚': 'shang', '商': 'shang', '伤': 'shang',
    '小': 'xiao', '晓': 'xiao', '校': 'xiao', '效': 'xiao',
    '快': 'kuai', '块': 'kuai', '筷': 'kuai', '会': 'kuai',
    '手': 'shou', '首': 'shou', '守': 'shou', '受': 'shou',
    '电': 'dian', '点': 'dian', '店': 'dian', '典': 'dian',
    '话': 'hua', '化': 'hua', '花': 'hua', '画': 'hua',
    '送': 'song', '宋': 'song', '松': 'song', '送': 'song',
    '出': 'chu', '初': 'chu', '除': 'chu', '处': 'chu',
    '入': 'ru', '如': 'ru', '乳': 'ru', '儒': 'ru',
    '开': 'kai', '凯': 'kai', '楷': 'kai', '慨': 'kai',
    '关': 'guan', '冠': 'guan', '官': 'guan', '观': 'guan',
    '网': 'wang', '往': 'wang', '望': 'wang', '王': 'wang',
    '零': 'ling', '一': 'yi', '二': 'er', '三': 'san', '四': 'si',
    '五': 'wu', '六': 'liu', '七': 'qi', '八': 'ba', '九': 'jiu', '十': 'shi',
    '了': 'le', '的': 'de', '是': 'shi', '在': 'zai', '有': 'you',
    '我': 'wo', '你': 'ni', '他': 'ta', '她': 'ta', '们': 'men',
    '不': 'bu', '就': 'jiu', '也': 'ye', '还': 'hai', '要': 'yao',
    '会': 'hui', '能': 'neng', '说': 'shuo', '看': 'kan', '做': 'zuo',
    '想': 'xiang', '去': 'qu', '用': 'yong', '对': 'dui', '自': 'zi',
    '子': 'zi', '日': 'ri', '月': 'yue', '年': 'nian', '时': 'shi',
    '到': 'dao', '人': 'ren', '天': 'tian', '地': 'di', '生': 'sheng',
}

# ---- 汉字 → 结构类型 ----
# 结构类型: 0=独体, 1=左右, 2=上下, 3=包围, 4=左中右, 5=上中下, 6=品字, 7=其他
STRUCT_TYPE_MAP = {
    '博': 1, '搏': 1, '薄': 2, '彩': 1, '采': 2, '赌': 1, '堵': 1,
    '码': 1, '马': 0, '证': 1, '正': 0, '办': 0, '卡': 2,
    '贷': 2, '代': 1, '款': 1, '宽': 2, '药': 2, '要': 2,
    '钱': 1, '前': 2, '信': 1, '心': 0, '微': 4, '危': 2,
    '元': 2, '原': 3, '万': 0, '一': 0, '大': 0,
    '加': 1, '家': 2, '号': 2, '好': 1, '来': 0,
    '下': 0, '夏': 2, '中': 0, '上': 0,
    '小': 0, '晓': 1, '快': 1, '块': 1,
    '手': 0, '首': 2, '电': 0, '话': 1,
    '送': 3, '宋': 2, '出': 0, '初': 1, '入': 0, '如': 1,
    '开': 0, '凯': 1, '关': 2, '冠': 2, '网': 3, '往': 1,
    '了': 0, '的': 1, '是': 2, '在': 3,
    '不': 0, '也': 0, '子': 0, '日': 0, '月': 0,
    '人': 0, '天': 0, '地': 1, '生': 0,
}

# ---- 汉字 → 部首（偏旁部件）----
RADICAL_MAP = {
    # 每个字展开为部首集合，模仿 IDS 表意文字描述序列
    '博': ['十', '甫', '寸'], '搏': ['扌', '甫', '寸'],
    '薄': ['艹', '氵', '甫', '寸'], '膊': ['月', '甫', '寸'],
    '彩': ['采', '彡'], '采': ['爫', '木'], '菜': ['艹', '采'],
    '赌': ['贝', '者'], '堵': ['土', '者'], '睹': ['目', '者'],
    '码': ['石', '马'], '马': ['马'], '妈': ['女', '马'],
    '证': ['讠', '正'], '正': ['正'], '政': ['正', '攵'],
    '办': ['力'], '半': ['丷', '十'], '伴': ['亻', '半'],
    '卡': ['上', '卜'], '咖': ['口', '加'],
    '贷': ['代', '贝'], '代': ['亻', '弋'], '款': ['士', '示', '欠'],
    '药': ['艹', '约'], '要': ['西', '女'], '腰': ['月', '要'],
    '钱': ['钅', '戋'], '前': ['䒑', '月', '刂'], '千': ['丿', '十'],
    '信': ['亻', '言'], '心': ['心'], '新': ['亲', '斤'],
    '微': ['彳', '山', '一', '几', '攵'], '危': ['⺈', '厂', '㔾'],
    '元': ['二', '儿'], '原': ['厂', '白', '小'], '圆': ['囗', '员'],
    '万': ['一', '勹'], '大': ['大'], '达': ['大', '辶'],
    '加': ['力', '口'], '家': ['宀', '豕'],
    '号': ['口', '丂'], '好': ['女', '子'],
    '来': ['一', '米'], '莱': ['艹', '来'],
    '下': ['一', '卜'], '夏': ['一', '自', '夂'],
    '中': ['口', '丨'], '上': ['卜', '一'],
    '小': ['小'], '晓': ['日', '尧'],
    '快': ['忄', '夬'], '块': ['土', '夬'],
    '手': ['手'], '首': ['䒑', '自'],
    '电': ['曰', '乚'], '话': ['讠', '舌'],
    '送': ['关', '辶'], '宋': ['宀', '木'],
    '出': ['屮', '凵'], '初': ['衤', '刀'],
    '入': ['入'], '如': ['女', '口'],
    '开': ['一', '廾'], '凯': ['岂', '几'],
    '关': ['丷', '天'], '冠': ['冖', '元', '寸'],
    '网': ['冂', '乂', '乂'], '往': ['彳', '主'],
    '零': ['雨', '令'], '贰': ['弋', '二', '贝'],
    '叁': ['厶', '三'], '肆': ['镸', '聿'],
    '伍': ['亻', '五'], '陆': ['阝', '击'],
    '柒': ['氵', '七', '木'], '捌': ['扌', '口', '力', '刂'],
    '玖': ['王', '久'], '拾': ['扌', '合'],
    '萬': ['艹', '禺'], '仟': ['亻', '千'],
    '佰': ['亻', '百'], '圓': ['囗', '員'],
    '証': ['言', '正'], '貸': ['代', '貝'],
    '欵': ['士', '欠'], '薬': ['艹', '楽'],
    '愽': ['忄', '專'], '採': ['扌', '采'],
    '賭': ['貝', '者'], '碼': ['石', '馬'],
    '薇': ['艹', '微'], '伩': ['亻', '文'],
    '伽': ['亻', '加'], '電': ['雨', '电'],
    '話': ['言', '舌'], '扌': ['扌'],
    # 常见字补充
    '了': ['了'], '的': ['白', '勺'], '是': ['日', '疋'],
    '在': ['亻', '土'], '有': ['月'], '我': ['手', '戈'],
    '你': ['亻', '尔'], '他': ['亻', '也'],
    '不': ['不'], '就': ['京', '尤'], '也': ['也'],
    '人': ['人'], '天': ['一', '大'], '地': ['土', '也'],
    '生': ['生'], '日': ['日'], '月': ['月'],
    '子': ['子'], '时': ['日', '寸'], '年': ['年'],
}

# ---- 笔画数（模拟）----
STROKE_COUNT_MAP = {}
for _ch, _rads in RADICAL_MAP.items():
    STROKE_COUNT_MAP[_ch] = min(30, len(_rads) * 3 + len(_ch))  # 粗略估计
# 直接覆盖常用字笔画数
STROKE_COUNT_ACCURATE = {
    '一': 1, '二': 2, '三': 3, '四': 5, '五': 4, '六': 4, '七': 2, '八': 2,
    '九': 2, '十': 2, '博': 12, '搏': 13, '彩': 11, '赌': 12, '码': 8,
    '马': 3, '证': 7, '办': 4, '卡': 5, '贷': 9, '款': 12, '药': 9,
    '钱': 10, '信': 9, '微': 13, '元': 4, '万': 3, '大': 3,
    '加': 5, '号': 5, '来': 7, '下': 3, '中': 4, '上': 3, '小': 3,
    '快': 7, '手': 4, '电': 5, '话': 8, '送': 9, '出': 5, '入': 2,
    '开': 4, '关': 6, '网': 6, '了': 2, '的': 8, '是': 9,
    '人': 2, '天': 4, '地': 6, '日': 4, '月': 4, '子': 3,
}
STROKE_COUNT_MAP.update(STROKE_COUNT_ACCURATE)


class CharacterGlyphDatabase:
    """
    中文字形数据库
    存储每个汉字的:
      - pinyin: 无调拼音
      - radicals: 偏旁部首列表
      - structure_type: 结构类型ID
      - stroke_count: 笔画数
    """
    def __init__(self):
        self.pinyin_map = PINYIN_MAP
        self.radical_map = RADICAL_MAP
        self.struct_map = STRUCT_TYPE_MAP
        self.stroke_map = STROKE_COUNT_MAP
        
        # 构建部件词表
        all_radicals = set()
        for rads in self.radical_map.values():
            for r in rads:
                all_radicals.add(r)
        self.radical_vocab = {r: i for i, r in enumerate(sorted(all_radicals))}
        self.num_radicals = len(self.radical_vocab)
        
        # 构建拼音词表
        all_pinyin = set(self.pinyin_map.values())
        self.pinyin_vocab = {p: i for i, p in enumerate(sorted(all_pinyin))}
        self.num_pinyin = len(self.pinyin_vocab)
        
        print(f'[GlyphDB] 部首词表: {self.num_radicals} 个, 拼音词表: {self.num_pinyin} 个')

    def get_glyph_features(self, text: str):
        """
        对输入文本，逐字提取字形特征矩阵
        
        返回:
            radical_feat: (len(text), num_radicals) 稀疏二值矩阵
            pinyin_idx:   (len(text),) 拼音ID向量
            struct_idx:   (len(text),) 结构类型ID
            stroke_feat:  (len(text), 1) 笔画数（归一化）
        """
        radical_rows, radical_cols, radical_vals = [], [], []
        pinyin_ids = []
        struct_ids = []
        stroke_vals = []
        
        for pos, ch in enumerate(text):
            # 部首特征（多热编码）
            rads = self.radical_map.get(ch, [])
            for r in rads:
                if r in self.radical_vocab:
                    radical_rows.append(pos)
                    radical_cols.append(self.radical_vocab[r])
                    radical_vals.append(1.0)
            
            # 拼音特征
            py = self.pinyin_map.get(ch, '_UNK_')
            if py not in self.pinyin_vocab:
                py = '_UNK_'
                if py not in self.pinyin_vocab:
                    self.pinyin_vocab[py] = len(self.pinyin_vocab)
            pinyin_ids.append(self.pinyin_vocab.get(py, 0))
            
            # 结构类型
            struct_ids.append(self.struct_map.get(ch, 7))  # 7=其他
            
            # 笔画数
            stroke_vals.append(self.stroke_map.get(ch, 5) / 30.0)  # 归一化
        
        # 构建稀疏部首矩阵
        from scipy.sparse import csr_matrix
        radical_feat = csr_matrix(
            (radical_vals, (radical_rows, radical_cols)),
            shape=(len(text), self.num_radicals)
        ) if radical_rows else csr_matrix((len(text), self.num_radicals))
        
        return {
            'radical_feat': radical_feat,
            'pinyin_idx': np.array(pinyin_ids, dtype=np.int64),
            'struct_idx': np.array(struct_ids, dtype=np.int64),
            'stroke_feat': np.array(stroke_vals, dtype=np.float32).reshape(-1, 1),
            'num_radicals': self.num_radicals,
            'num_pinyin': self.num_pinyin,
        }

    def compute_glyph_distance(self, char_a: str, char_b: str) -> float:
        """计算两个汉字的字形相似度 (0=完全不同, 1=完全相同)"""
        if char_a == char_b:
            return 1.0
        
        rads_a = set(self.radical_map.get(char_a, []))
        rads_b = set(self.radical_map.get(char_b, []))
        
        if not rads_a and not rads_b:
            return 0.0
        if not rads_a or not rads_b:
            return 0.0
        
        # Jaccard 相似度（共享部首 / 总部首）
        intersection = len(rads_a & rads_b)
        union = len(rads_a | rads_b)
        rad_sim = intersection / union if union > 0 else 0.0
        
        # 拼音相似度
        py_a = self.pinyin_map.get(char_a, '')
        py_b = self.pinyin_map.get(char_b, '')
        py_sim = 1.0 if (py_a == py_b and py_a != '') else 0.0
        
        # 结构相似度
        struct_a = self.struct_map.get(char_a, 7)
        struct_b = self.struct_map.get(char_b, 7)
        struct_sim = 1.0 if struct_a == struct_b else 0.0
        
        # 加权融合
        return 0.5 * rad_sim + 0.3 * py_sim + 0.2 * struct_sim


# ==================== 第二部分：GlyphEmbedding 网络 ====================

class GlyphEmbedding:
    """
    字形嵌入模块 (PyTorch)
    将中文字符的部首/拼音/结构/笔画四维特征融合成一个固定维度的向量。
    
    输入:  字符的 glyph_features 字典
    输出:  (batch, seq_len, glyph_dim) 的字形嵌入矩阵
    """
    def __init__(self, num_radicals: int, num_pinyin: int, 
                 glyph_dim: int = 128, pinyin_embed_dim: int = 32, 
                 struct_embed_dim: int = 16):
        import torch
        import torch.nn as nn
        
        self.num_radicals = num_radicals
        self.num_pinyin = num_pinyin
        self.glyph_dim = glyph_dim
        
        # 部首 → 稠密向量 (用线性层代替嵌入，处理多热输入)
        self.radical_projector = nn.Linear(num_radicals, glyph_dim)
        
        # 拼音 Embedding
        self.pinyin_embed = nn.Embedding(num_pinyin, pinyin_embed_dim)
        
        # 结构类型 Embedding (0-7 共 8 种结构)
        self.struct_embed = nn.Embedding(8, struct_embed_dim)
        
        # 笔画数 projector
        self.stroke_projector = nn.Linear(1, 16)
        
        # 融合层
        fusion_input_dim = glyph_dim + pinyin_embed_dim + struct_embed_dim + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, glyph_dim),
            nn.LayerNorm(glyph_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(glyph_dim, glyph_dim),
            nn.LayerNorm(glyph_dim),
        )
        
    def forward(self, glyph_features: dict):
        """
        Args:
            glyph_features: 包含
                - radical_feat: (batch, seq_len, num_radicals) 的稀疏张量
                - pinyin_idx: (batch, seq_len) int
                - struct_idx: (batch, seq_len) int
                - stroke_feat: (batch, seq_len, 1) float
        Returns:
            glyph_embed: (batch, seq_len, glyph_dim)
        """
        import torch
        
        # 部首投影
        rad = glyph_features['radical_feat'].to_dense() if hasattr(glyph_features['radical_feat'], 'to_dense') else glyph_features['radical_feat']
        rad_embed = self.radical_projector(rad.float())  # (B, S, glyph_dim)
        
        # 拼音嵌入
        py = glyph_features['pinyin_idx']
        py_embed = self.pinyin_embed(py)  # (B, S, pinyin_embed_dim)
        
        # 结构嵌入
        st = glyph_features['struct_idx']
        st_embed = self.struct_embed(st)  # (B, S, struct_embed_dim)
        
        # 笔画投影
        sk = glyph_features['stroke_feat']
        sk_embed = self.stroke_projector(sk)  # (B, S, 16)
        
        # 拼接并融合
        concat = torch.cat([rad_embed, py_embed, st_embed, sk_embed], dim=-1)
        glyph_embed = self.fusion(concat)
        
        return glyph_embed


# ==================== 第三部分：GCA-Net 分类器 ====================

class GCANet:
    """
    Glyph-Contrastive Alignment Network
    
    训练流程:
      1. 构建 GlyphDatabase，提取所有训练文本的字形特征
      2. 用 GlyphEmbedding 编码字形
      3. 与 TF-IDF 特征拼接，输入分类器
      4. 对比学习损失：拉近形近字嵌入、推远无关字嵌入
    """
    def __init__(self, name: str = 'GCA-Net'):
        self.name = name
        self.glyph_db = CharacterGlyphDatabase()
        self.glyph_embed = None
        self.classifier = None
        self.device = None
        self.tfidf_dim = None
        self.num_classes = None
    
    def fit(self, X_tfidf, y, texts, epochs=15, batch_size=64, lr=1e-3, 
            glyph_weight=0.3, contrast_weight=0.1):
        """
        训练 GCA-Net
        
        Args:
            X_tfidf: TF-IDF 特征矩阵 (N, D) 稀疏或稠密
            y: 标签 (N,)
            texts: 原始文本列表 (对 TF-IDF 分词后的文本重新清洗)
            epochs: 训练轮数
            glyph_weight: 字形特征权重
            contrast_weight: 对比学习损失权重
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        
        self.tfidf_dim = X_tfidf.shape[1]
        self.num_classes = len(set(y))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ---- 准备数据 ----
        # TF-IDF
        if hasattr(X_tfidf, 'toarray'):
            X_tfidf_dense = X_tfidf.toarray()
        else:
            X_tfidf_dense = X_tfidf
        
        # 字形特征（批量提取）
        print(f'  [GCA-Net] 提取字形特征...')
        # 逐样本构建，限制序列长度
        max_seq = 256
        N = len(texts)
        radical_batches = []
        pinyin_batches = []
        struct_batches = []
        stroke_batches = []
        
        for text in texts[:N]:
            clean_text = re.sub(r'\s+', '', str(text))[:max_seq]
            if not clean_text:
                clean_text = ' '
            
            gf = self.glyph_db.get_glyph_features(clean_text)
            seq_len = len(clean_text)
            
            # 部首特征转密集
            rad_dense = np.zeros((max_seq, self.glyph_db.num_radicals), dtype=np.float32)
            if hasattr(gf['radical_feat'], 'toarray'):
                r = gf['radical_feat'].toarray()
            else:
                r = gf['radical_feat']
            rad_dense[:min(seq_len, max_seq), :] = r[:max_seq, :] if r.shape[0] > 0 else rad_dense
            radical_batches.append(rad_dense)
            
            # 拼音
            py_ids = np.zeros(max_seq, dtype=np.int64)
            py_ids[:min(seq_len, max_seq)] = gf['pinyin_idx'][:max_seq]
            pinyin_batches.append(py_ids)
            
            # 结构
            st_ids = np.full(max_seq, 7, dtype=np.int64)
            st_ids[:min(seq_len, max_seq)] = gf['struct_idx'][:max_seq]
            struct_batches.append(st_ids)
            
            # 笔画
            sk = np.zeros((max_seq, 1), dtype=np.float32)
            sk[:min(seq_len, max_seq), 0] = gf['stroke_feat'][:max_seq, 0]
            stroke_batches.append(sk)
        
        radical_feat = np.stack(radical_batches, axis=0)
        pinyin_idx = np.stack(pinyin_batches, axis=0)
        struct_idx = np.stack(struct_batches, axis=0)
        stroke_feat = np.stack(stroke_batches, axis=0)
        
        X_tfidf_tensor = torch.FloatTensor(X_tfidf_dense[:N])
        y_tensor = torch.LongTensor(np.array(y)[:N])
        radical_tensor = torch.FloatTensor(radical_feat)
        pinyin_tensor = torch.LongTensor(pinyin_idx)
        struct_tensor = torch.LongTensor(struct_idx)
        stroke_tensor = torch.FloatTensor(stroke_feat)
        
        dataset = TensorDataset(X_tfidf_tensor, radical_tensor, pinyin_tensor, 
                                struct_tensor, stroke_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # ---- 构建模型 ----
        glyph_dim = 128
        self.glyph_embed = GlyphEmbedding(
            num_radicals=self.glyph_db.num_radicals,
            num_pinyin=self.glyph_db.num_pinyin,
            glyph_dim=glyph_dim
        ).to(self.device)
        
        # 分类器（TF-IDF + 字形特征融合）
        self.classifier = nn.Sequential(
            nn.Linear(self.tfidf_dim + glyph_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        ).to(self.device)
        
        # 对比学习用的字形投影头
        self.glyph_projection = nn.Sequential(
            nn.Linear(glyph_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(self.device)
        
        # ---- 优化器 ----
        all_params = (list(self.glyph_embed.parameters()) + 
                      list(self.classifier.parameters()) +
                      list(self.glyph_projection.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # ---- 训练循环 ----
        print(f'  [GCA-Net] 开始训练, 设备: {self.device}')
        for epoch in range(epochs):
            total_loss = 0
            total_ce = 0
            total_contrast = 0
            
            self.glyph_embed.train()
            self.classifier.train()
            self.glyph_projection.train()
            
            for bx_tfidf, bx_rad, bx_py, bx_st, bx_sk, by in loader:
                bx_tfidf = bx_tfidf.to(self.device)
                bx_rad = bx_rad.to(self.device)
                bx_py = bx_py.to(self.device)
                bx_st = bx_st.to(self.device)
                bx_sk = bx_sk.to(self.device)
                by = by.to(self.device)
                
                optimizer.zero_grad()
                
                # 字形编码
                glyph_features = {
                    'radical_feat': bx_rad, 'pinyin_idx': bx_py,
                    'struct_idx': bx_st, 'stroke_feat': bx_sk,
                }
                glyph_seq = self.glyph_embed.forward(glyph_features)  # (B, S, glyph_dim)
                
                # 全局池化 (mean pooling over sequence)
                glyph_pooled = glyph_seq.mean(dim=1)  # (B, glyph_dim)
                
                # 与 TF-IDF 拼接
                combined = torch.cat([bx_tfidf * (1 - glyph_weight), 
                                      glyph_pooled * glyph_weight], dim=-1)
                
                # 分类损失
                logits = self.classifier(combined)
                ce_loss = criterion(logits, by)
                
                # 对比学习损失 (InfoNCE)
                # 在 batch 内，同一 batch 的不同样本的字形特征应该具有区分性
                # 这里我们用简化版：最大化 batch 内字形特征的方差
                glyph_proj = self.glyph_projection(glyph_pooled)
                glyph_proj = F.normalize(glyph_proj, dim=-1)
                
                # 计算 pairwise 相似度矩阵
                sim_matrix = glyph_proj @ glyph_proj.T  # (B, B)
                
                # 标签相同的视为正样本对，标签不同的视为负样本对
                label_eq = (by.unsqueeze(0) == by.unsqueeze(1)).float()  # (B, B)
                label_ne = 1 - label_eq
                
                # 对比损失：正样本对相似度高，负样本对相似度低
                pos_sim = (sim_matrix * label_eq).sum() / (label_eq.sum() + 1e-8)
                neg_sim = (sim_matrix * label_ne).sum() / (label_ne.sum() + 1e-8)
                contrast_loss = -pos_sim + neg_sim
                
                # 总损失
                loss = ce_loss + contrast_weight * contrast_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_contrast += contrast_loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'  [{self.name}] Epoch {epoch+1}/{epochs}, '
                      f'Loss: {total_loss/len(loader):.4f}, '
                      f'CE: {total_ce/len(loader):.4f}, '
                      f'Contrast: {total_contrast/len(loader):.4f}')
        
        return self
    
    def _extract_batch_glyph_features(self, texts):
        """批量提取字形特征"""
        import torch
        max_seq = 256
        batch_size = len(texts)
        
        radical_batches = []
        pinyin_ids = []
        struct_ids = []
        stroke_vals = []
        
        for text in texts:
            clean = re.sub(r'\s+', '', str(text))[:max_seq]
            if not clean:
                clean = ' '
            gf = self.glyph_db.get_glyph_features(clean)
            seq_len = len(clean)
            
            rad_dense = np.zeros((max_seq, self.glyph_db.num_radicals), dtype=np.float32)
            r = gf['radical_feat']
            if hasattr(r, 'toarray'):
                r = r.toarray()
            rad_dense[:min(seq_len, max_seq), :] = r[:max_seq, :] if r.shape[0] > 0 else rad_dense
            radical_batches.append(rad_dense)
            
            py_ids = np.zeros(max_seq, dtype=np.int64)
            py_ids[:min(seq_len, max_seq)] = gf['pinyin_idx'][:max_seq]
            pinyin_ids.append(py_ids)
            
            st_ids = np.full(max_seq, 7, dtype=np.int64)
            st_ids[:min(seq_len, max_seq)] = gf['struct_idx'][:max_seq]
            struct_ids.append(st_ids)
            
            sk = np.zeros((max_seq, 1), dtype=np.float32)
            sk[:min(seq_len, max_seq), 0] = gf['stroke_feat'][:max_seq, 0]
            stroke_vals.append(sk)
        
        return {
            'radical_feat': torch.FloatTensor(np.stack(radical_batches, axis=0)).to(self.device),
            'pinyin_idx': torch.LongTensor(np.stack(pinyin_ids, axis=0)).to(self.device),
            'struct_idx': torch.LongTensor(np.stack(struct_ids, axis=0)).to(self.device),
            'stroke_feat': torch.FloatTensor(np.stack(stroke_vals, axis=0)).to(self.device),
        }
    
    def predict(self, X):
        """预测标签"""
        import torch
        self.glyph_embed.eval()
        self.classifier.eval()
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray() if hasattr(X, 'toarray') else X
        else:
            X_dense = X
        if isinstance(X_dense, np.ndarray):
            X_dense = torch.FloatTensor(X_dense).to(self.device)
        
        # 无文本信息时（仅 TF-IDF 输入），用零填充字形特征
        glyph_dummy = torch.zeros(X_dense.shape[0], 128).to(self.device)
        combined = torch.cat([X_dense * 0.7, glyph_dummy * 0.3], dim=-1)
        
        with torch.no_grad():
            logits = self.classifier(combined)
            pred = logits.argmax(dim=1).cpu().numpy()
        return pred
    
    def predict_with_texts(self, X_tfidf, texts):
        """带文本的预测（可以使用字形特征）"""
        import torch
        self.glyph_embed.eval()
        self.classifier.eval()
        
        if hasattr(X_tfidf, 'toarray'):
            X_dense = X_tfidf.toarray()
        else:
            X_dense = X_tfidf
        X_tensor = torch.FloatTensor(X_dense).to(self.device)
        
        # 提取字形特征
        gf = self._extract_batch_glyph_features(texts)
        with torch.no_grad():
            glyph_seq = self.glyph_embed.forward(gf)
            glyph_pooled = glyph_seq.mean(dim=1)
        
        combined = torch.cat([X_tensor * 0.7, glyph_pooled * 0.3], dim=-1)
        
        with torch.no_grad():
            logits = self.classifier(combined)
            pred = logits.argmax(dim=1).cpu().numpy()
        return pred
    
    def predict_proba(self, X):
        """预测概率"""
        import torch
        self.glyph_embed.eval()
        self.classifier.eval()
        
        if hasattr(X, 'toarray'):
            X_dense = X.toarray() if hasattr(X, 'toarray') else X
        else:
            X_dense = X
        if isinstance(X_dense, np.ndarray):
            X_dense = torch.FloatTensor(X_dense).to(self.device)
        
        glyph_dummy = torch.zeros(X_dense.shape[0], 128).to(self.device)
        combined = torch.cat([X_dense * 0.7, glyph_dummy * 0.3], dim=-1)
        
        with torch.no_grad():
            logits = self.classifier(combined)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba
    
    def predict_proba_with_texts(self, X_tfidf, texts):
        """带文本的预测概率（使用字形特征）"""
        import torch
        self.glyph_embed.eval()
        self.classifier.eval()
        
        if hasattr(X_tfidf, 'toarray'):
            X_dense = X_tfidf.toarray()
        else:
            X_dense = X_tfidf
        X_tensor = torch.FloatTensor(X_dense).to(self.device)
        
        gf = self._extract_batch_glyph_features(texts)
        with torch.no_grad():
            glyph_seq = self.glyph_embed.forward(gf)
            glyph_pooled = glyph_seq.mean(dim=1)
        
        combined = torch.cat([X_tensor * 0.7, glyph_pooled * 0.3], dim=-1)
        
        with torch.no_grad():
            logits = self.classifier(combined)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba
    
    def save(self, path: str):
        import torch
        torch.save({
            'glyph_embed_state': self.glyph_embed.state_dict(),
            'classifier_state': self.classifier.state_dict(),
            'projection_state': self.glyph_projection.state_dict(),
            'tfidf_dim': self.tfidf_dim,
            'num_classes': self.num_classes,
            'num_radicals': self.glyph_db.num_radicals,
            'num_pinyin': self.glyph_db.num_pinyin,
        }, path)
        print(f'  [{self.name}] 模型已保存至: {path}')
    
    def load(self, path: str):
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        self.tfidf_dim = checkpoint['tfidf_dim']
        self.num_classes = checkpoint['num_classes']
        self.glyph_db = CharacterGlyphDatabase()
        
        self.glyph_embed = GlyphEmbedding(
            num_radicals=checkpoint['num_radicals'],
            num_pinyin=checkpoint['num_pinyin'],
            glyph_dim=128
        )
        self.glyph_embed.load_state_dict(checkpoint['glyph_embed_state'])
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.tfidf_dim + 128, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, self.num_classes)
        )
        self.classifier.load_state_dict(checkpoint['classifier_state'])
        
        self.glyph_projection = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        self.glyph_projection.load_state_dict(checkpoint['projection_state'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.glyph_embed.to(self.device)
        self.classifier.to(self.device)
        self.glyph_projection.to(self.device)
        print(f'  [{self.name}] 模型已加载自: {path}')


# ==================== 第四部分：字形相似度分析工具 ====================

def analyze_glyph_similarity():
    """分析形近字对在字形空间中的相似度"""
    db = CharacterGlyphDatabase()
    
    # 对抗样本中使用的形近/音近字对
    pairs = [
        ('博', '搏'), ('彩', '採'), ('赌', '賭'), ('码', '碼'),
        ('证', '証'), ('贷', '貸'), ('款', '欵'), ('药', '薬'),
        ('微', '薇'), ('信', '伩'), ('加', '伽'), ('电', '電'),
        ('话', '話'), ('手', '扌'), ('元', '圆'),
        # 同音字对
        ('博', '搏'), ('彩', '采'), ('赌', '堵'), ('码', '马'),
        ('证', '正'), ('贷', '代'), ('药', '要'), ('钱', '前'),
        ('下', '夏'), ('中', '忠'), ('小', '晓'),
    ]
    
    print(f'\n{"="*60}')
    print(f'  字形相似度分析')
    print(f'{"="*60}')
    print(f'  {"字对":<10s} {"部首Jaccard":>12s} {"拼音":>6s} {"结构":>6s} {"综合相似度":>10s}')
    print(f'  {"-"*50}')
    
    for a, b in pairs:
        sim = db.compute_glyph_distance(a, b)
        rads_a = set(db.radical_map.get(a, []))
        rads_b = set(db.radical_map.get(b, []))
        jc = len(rads_a & rads_b) / max(1, len(rads_a | rads_b))
        py_same = '✓' if db.pinyin_map.get(a) == db.pinyin_map.get(b) else '✗'
        st_same = '✓' if db.struct_map.get(a) == db.struct_map.get(b) else '✗'
        print(f'  {a}→{b:<7s} {jc:>12.3f} {py_same:>6s} {st_same:>6s} {sim:>10.3f}')
    
    print()
    print('  解释: 部首Jaccard > 0 表示共享偏旁部件（视觉相似）')
    print('        拼音相同 → 同音字（语音相似）')
    print('        传统BOW/BERT无法建模这些相似性，GCA-Net可以')


if __name__ == '__main__':
    # 测试字形数据库
    db = CharacterGlyphDatabase()
    print(f'\n部首词表大小: {db.num_radicals}')
    print(f'拼音词表大小: {db.num_pinyin}')
    
    # 测试字形特征提取
    test_text = '专业办证，信用卡提现'
    gf = db.get_glyph_features(test_text)
    print(f'\n测试文本: {test_text}')
    print(f'  字形部首矩阵维度: {gf["radical_feat"].shape}')
    print(f'  拼音索引: {gf["pinyin_idx"]}')
    print(f'  结构类型: {gf["struct_idx"]}')
    print(f'  笔画特征: {gf["stroke_feat"].flatten()}')
    
    # 相似度分析
    analyze_glyph_similarity()
