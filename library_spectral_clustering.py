"""
高校图书馆借阅记录分析：读者谱聚类
- 构建读者-图书二部图
- 基于学科分类计算Jaccard相似度
- 谱聚类分组 + 特征值间隙确定聚类数
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
import networkx as nx
from networkx.algorithms import bipartite
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 模拟三个月借阅记录
# ============================================================

SUBJECT_CATEGORIES = {
    "计算机": [
        "数据结构与算法", "机器学习实战", "深度学习", "计算机网络",
        "操作系统概论", "Python编程", "数据库系统", "编译原理",
        "人工智能导论", "软件工程"
    ],
    "文学": [
        "红楼梦", "百年孤独", "围城", "活着", "平凡的世界",
        "挪威的森林", "追风筝的人", "三体", "白鹿原", "人间失格"
    ],
    "历史": [
        "万历十五年", "人类简史", "明朝那些事儿", "全球通史",
        "史记选读", "中国近代史", "罗马帝国衰亡史", "资治通鉴",
        "丝绸之路", "枪炮病菌与钢铁"
    ],
    "经济": [
        "国富论", "经济学原理", "资本论", "货币金融学",
        "博弈论", "行为经济学", "宏观经济学", "微观经济学",
        "金融学", "计量经济学"
    ],
    "哲学": [
        "西方哲学史", "存在与时间", "理想国", "纯粹理性批判",
        "道德经", "论语", "沉思录", "查拉图斯特拉如是说",
        "中国哲学简史", "苏菲的世界"
    ],
    "数学": [
        "高等数学", "线性代数", "概率论与数理统计", "数学分析",
        "离散数学", "拓扑学导引", "微分方程", "数值分析",
        "实变函数", "泛函分析"
    ],
}

BOOK_TO_SUBJECT = {}
ALL_BOOKS = []
for subject, books in SUBJECT_CATEGORIES.items():
    for book in books:
        BOOK_TO_SUBJECT[book] = subject
        ALL_BOOKS.append(book)

NUM_READERS = 200
MONTHS = ["2025-09", "2025-10", "2025-11"]

# 为每个读者定义偏好（模拟真实借阅行为）
READER_PROFILES = {
    "技术型": {"计算机": 0.40, "数学": 0.25, "经济": 0.10, "文学": 0.10, "历史": 0.10, "哲学": 0.05},
    "文艺型": {"文学": 0.40, "哲学": 0.25, "历史": 0.15, "经济": 0.05, "计算机": 0.05, "数学": 0.10},
    "社科型": {"历史": 0.30, "经济": 0.25, "哲学": 0.20, "文学": 0.15, "计算机": 0.05, "数学": 0.05},
    "理工型": {"数学": 0.35, "计算机": 0.30, "经济": 0.10, "哲学": 0.05, "文学": 0.10, "历史": 0.10},
    "杂食型": {"计算机": 0.17, "文学": 0.17, "历史": 0.17, "经济": 0.17, "哲学": 0.16, "数学": 0.16},
}

profile_names = list(READER_PROFILES.keys())
profile_weights = [0.25, 0.20, 0.20, 0.20, 0.15]


def generate_borrowing_records():
    records = []
    reader_profiles_assigned = {}

    for reader_id in range(1, NUM_READERS + 1):
        profile_name = np.random.choice(profile_names, p=profile_weights)
        reader_profiles_assigned[f"R{reader_id:03d}"] = profile_name
        probs = READER_PROFILES[profile_name]

        subjects = list(probs.keys())
        subject_probs = [probs[s] for s in subjects]

        num_borrows = np.random.randint(5, 20)
        for _ in range(num_borrows):
            subject = np.random.choice(subjects, p=subject_probs)
            book = np.random.choice(SUBJECT_CATEGORIES[subject])
            month = np.random.choice(MONTHS)
            day = np.random.randint(1, 28)
            records.append({
                "reader_id": f"R{reader_id:03d}",
                "book": book,
                "subject": subject,
                "date": f"{month}-{day:02d}",
            })

    return pd.DataFrame(records), reader_profiles_assigned


print("=" * 60)
print("高校图书馆借阅记录谱聚类分析")
print("=" * 60)

print("\n[1] 生成三个月模拟借阅记录...")
df_records, reader_profiles_assigned = generate_borrowing_records()
print(f"    读者数量: {NUM_READERS}")
print(f"    借阅记录: {len(df_records)} 条")
print(f"    时间范围: {MONTHS[0]} ~ {MONTHS[-1]}")
print(f"    学科类别: {list(SUBJECT_CATEGORIES.keys())}")
print(f"\n    借阅记录示例:")
print(df_records.head(10).to_string(index=False))

# ============================================================
# 2. 构建读者-图书二部图
# ============================================================

print("\n[2] 构建读者-图书二部图...")
B = nx.Graph()

readers = df_records["reader_id"].unique()
books = df_records["book"].unique()

B.add_nodes_from(readers, bipartite=0)
B.add_nodes_from(books, bipartite=1)

for _, row in df_records.iterrows():
    if B.has_edge(row["reader_id"], row["book"]):
        B[row["reader_id"]][row["book"]]["weight"] += 1
    else:
        B.add_edge(row["reader_id"], row["book"], weight=1)

print(f"    二部图节点数: {B.number_of_nodes()} (读者: {len(readers)}, 图书: {len(books)})")
print(f"    二部图边数: {B.number_of_edges()}")

# ============================================================
# 3. 计算读者学科借阅向量 & Jaccard相似度
# ============================================================

print("\n[3] 计算读者间Jaccard相似度...")

subjects_list = sorted(SUBJECT_CATEGORIES.keys())
subject_idx = {s: i for i, s in enumerate(subjects_list)}

# 每个读者的学科借阅次数向量
reader_subject_counts = defaultdict(lambda: np.zeros(len(subjects_list)))
for _, row in df_records.iterrows():
    reader_subject_counts[row["reader_id"]][subject_idx[row["subject"]]] += 1

reader_ids = sorted(reader_subject_counts.keys())
n = len(reader_ids)


def jaccard_on_categories(vec_a, vec_b):
    """标准Jaccard系数：交集元素数 / 并集元素数（基于借阅过的学科集合）"""
    set_a = set(np.where(vec_a > 0)[0])
    set_b = set(np.where(vec_b > 0)[0])
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# 构建相似度矩阵
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        sim = jaccard_on_categories(
            reader_subject_counts[reader_ids[i]],
            reader_subject_counts[reader_ids[j]]
        )
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

np.fill_diagonal(similarity_matrix, 1.0)

print(f"    相似度矩阵大小: {similarity_matrix.shape}")
print(f"    平均Jaccard相似度: {similarity_matrix[np.triu_indices(n, k=1)].mean():.4f}")
print(f"    最大Jaccard相似度: {similarity_matrix[np.triu_indices(n, k=1)].max():.4f}")

# ============================================================
# 4. 谱聚类 + 特征值间隙确定聚类数
# ============================================================

print("\n[4] 谱聚类分析...")

# 构建拉普拉斯矩阵
W = similarity_matrix.copy()
np.fill_diagonal(W, 0)
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt  # 归一化拉普拉斯

# 计算前10个最小特征值
num_eig = 10
eigenvalues, eigenvectors = eigsh(L_norm, k=num_eig, which='SM')

# 排序
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"    前{num_eig}个特征值: {np.round(eigenvalues, 6)}")

# 特征值间隙
gaps = np.diff(eigenvalues)
print(f"    特征值间隙: {np.round(gaps, 6)}")

# 在2-10范围内选择最大间隙对应的聚类数
candidate_gaps = {k: gaps[k - 1] for k in range(2, 11) if k - 1 < len(gaps)}
optimal_k = max(candidate_gaps, key=candidate_gaps.get)
print(f"    候选聚类数间隙 (k=2~10): {candidate_gaps}")
print(f"    最优聚类数(特征值间隙): k = {optimal_k}")

# 使用前k个特征向量进行KMeans
H = eigenvectors[:, :optimal_k]
# 行归一化
H_norm = H / np.linalg.norm(H, axis=1, keepdims=True)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
labels = kmeans.fit_predict(H_norm)

print(f"    聚类完成，各组人数:")
for c in range(optimal_k):
    count = np.sum(labels == c)
    print(f"      组{c + 1}: {count} 人")

# ============================================================
# 5. 分析每组特征并起昵称
# ============================================================

print("\n[5] 聚类结果分析")
print("=" * 60)

NICKNAMES = {
    ("计算机", "数学"): "代码骑士",
    ("计算机", "数学", "经济"): "全栈极客",
    ("文学", "哲学"): "文艺青年",
    ("文学", "哲学", "历史"): "人文行者",
    ("历史", "经济"): "纵横家",
    ("历史", "经济", "哲学"): "思想者联盟",
    ("数学", "计算机"): "算法达人",
    ("数学", "计算机", "经济"): "量化先锋",
    ("经济", "历史"): "经世致用派",
    ("经济", "哲学"): "理性思辨者",
    ("哲学", "文学"): "精神漫游者",
    ("哲学", "历史", "文学"): "智慧探索者",
}

DEFAULT_NICKNAMES = ["求知者", "博览群书", "跨界达人", "学海无涯", "书山有路", "探索者"]


def get_nickname(top_subjects):
    # 先查3学科组合
    key3 = tuple(top_subjects[:3])
    if key3 in NICKNAMES:
        return NICKNAMES[key3]
    # 再查2学科组合
    key = tuple(top_subjects[:2])
    if key in NICKNAMES:
        return NICKNAMES[key]
    # 尝试反转2学科组合
    key_rev = tuple(reversed(top_subjects[:2]))
    if key_rev in NICKNAMES:
        return NICKNAMES[key_rev]
    # 基于主要学科生成昵称
    subject_nicknames = {
        "计算机": "数字先锋",
        "文学": "墨香书生",
        "历史": "鉴古知今",
        "经济": "经世济民",
        "哲学": "爱智求真",
        "数学": "数理精英",
    }
    return subject_nicknames.get(top_subjects[0], DEFAULT_NICKNAMES[0])


results = []

for c in range(optimal_k):
    cluster_readers = [reader_ids[i] for i in range(n) if labels[i] == c]
    cluster_records = df_records[df_records["reader_id"].isin(cluster_readers)]

    subject_counts = cluster_records["subject"].value_counts()
    top3 = subject_counts.head(3)
    top3_subjects = top3.index.tolist()

    nickname = get_nickname(top3_subjects)

    total_borrows = len(cluster_records)
    results.append({
        "group": c + 1,
        "nickname": nickname,
        "size": len(cluster_readers),
        "top3_subjects": top3_subjects,
        "top3_counts": top3.values.tolist(),
        "total_borrows": total_borrows,
    })

    print(f"\n┌─────────────────────────────────────────────────────────")
    print(f"│ 组 {c + 1}: 「{nickname}」")
    print(f"├─────────────────────────────────────────────────────────")
    print(f"│ 人数: {len(cluster_readers)} 人")
    print(f"│ 总借阅量: {total_borrows} 次")
    print(f"│ 最常借阅学科 TOP-3:")
    for rank, (subj, cnt) in enumerate(zip(top3_subjects, top3.values), 1):
        pct = cnt / total_borrows * 100
        bar = "█" * int(pct / 2)
        print(f"│   {rank}. {subj:　<4} : {cnt:3d} 次 ({pct:5.1f}%) {bar}")
    print(f"│ 代表读者: {', '.join(cluster_readers[:5])}...")
    print(f"└─────────────────────────────────────────────────────────")

# ============================================================
# 6. 汇总表
# ============================================================

print("\n\n" + "=" * 60)
print("汇总表")
print("=" * 60)
print(f"{'组号':<4} {'昵称':<10} {'人数':<6} {'TOP-1':<6} {'TOP-2':<6} {'TOP-3':<6}")
print("-" * 60)
for r in results:
    subjects = r["top3_subjects"]
    print(f" {r['group']:<3} {r['nickname']:<10} {r['size']:<6} {subjects[0]:<6} {subjects[1]:<6} {subjects[2]:<6}")

print("\n" + "=" * 60)
print("分析完成")
print(f"  - 聚类方法: 归一化谱聚类 (Normalized Spectral Clustering)")
print(f"  - 相似度度量: 基于学科借阅频次的Jaccard系数")
print(f"  - 聚类数确定: 特征值间隙法 (Eigengap Heuristic)")
print(f"  - 最终聚类数: {optimal_k}")
print("=" * 60)
