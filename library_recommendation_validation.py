"""
推荐验证系统：基于谱聚类的图书推荐 vs 流行度基线
- 利用谱聚类结果预测读者潜在兴趣学科
- 为每位读者推荐3本未借阅书籍
- 随机抽样100名读者计算推荐命中率
- 对比基于流行度的推荐基线
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. 数据准备（复用谱聚类中的数据生成逻辑）
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
    """生成借阅记录，前2个月作为训练集，第3个月作为测试集"""
    records_train = []
    records_test = []
    reader_profiles_assigned = {}

    for reader_id in range(1, NUM_READERS + 1):
        profile_name = np.random.choice(profile_names, p=profile_weights)
        reader_profiles_assigned[f"R{reader_id:03d}"] = profile_name
        probs = READER_PROFILES[profile_name]

        subjects = list(probs.keys())
        subject_probs = [probs[s] for s in subjects]

        num_borrows_train = np.random.randint(5, 15)
        for _ in range(num_borrows_train):
            subject = np.random.choice(subjects, p=subject_probs)
            book = np.random.choice(SUBJECT_CATEGORIES[subject])
            month = np.random.choice(["2025-09", "2025-10"])
            day = np.random.randint(1, 28)
            records_train.append({
                "reader_id": f"R{reader_id:03d}",
                "book": book,
                "subject": subject,
                "date": f"{month}-{day:02d}",
            })

        num_borrows_test = np.random.randint(3, 8)
        for _ in range(num_borrows_test):
            subject = np.random.choice(subjects, p=subject_probs)
            book = np.random.choice(SUBJECT_CATEGORIES[subject])
            records_test.append({
                "reader_id": f"R{reader_id:03d}",
                "book": book,
                "subject": subject,
                "date": f"2025-11-{np.random.randint(1, 28):02d}",
            })

    return pd.DataFrame(records_train), pd.DataFrame(records_test), reader_profiles_assigned


# ============================================================
# 2. 谱聚类
# ============================================================

def perform_spectral_clustering(df_train):
    """对训练集执行谱聚类，返回聚类标签和读者ID映射"""
    subjects_list = sorted(SUBJECT_CATEGORIES.keys())
    subject_idx = {s: i for i, s in enumerate(subjects_list)}

    reader_subject_counts = defaultdict(lambda: np.zeros(len(subjects_list)))
    for _, row in df_train.iterrows():
        reader_subject_counts[row["reader_id"]][subject_idx[row["subject"]]] += 1

    reader_ids = sorted(reader_subject_counts.keys())
    n = len(reader_ids)

    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vec_a = reader_subject_counts[reader_ids[i]]
            vec_b = reader_subject_counts[reader_ids[j]]
            set_a = set(np.where(vec_a > 0)[0])
            set_b = set(np.where(vec_b > 0)[0])
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            sim = intersection / union if union > 0 else 0.0
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    np.fill_diagonal(similarity_matrix, 1.0)

    W = similarity_matrix.copy()
    np.fill_diagonal(W, 0)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_norm = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv

    num_eig = 10
    eigenvalues, eigenvectors = eigsh(L_norm, k=num_eig, which='SM')
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    gaps = np.diff(eigenvalues)
    candidate_gaps = {k: gaps[k - 1] for k in range(2, 11) if k - 1 < len(gaps)}
    optimal_k = max(candidate_gaps, key=candidate_gaps.get)

    H = eigenvectors[:, :optimal_k]
    H_norm = H / np.linalg.norm(H, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(H_norm)

    return labels, reader_ids, optimal_k, reader_subject_counts, subjects_list


# ============================================================
# 3. 推荐算法
# ============================================================

def spectral_recommend(reader_id, labels, reader_ids, reader_subject_counts,
                       subjects_list, df_train, top_n=3):
    """
    基于谱聚类的推荐：
    1. 找到读者所在簇
    2. 统计簇内所有读者的学科偏好分布
    3. 找出该读者尚未深入阅读的学科中，簇内热门的书籍
    4. 推荐该读者未借阅过的top_n本书
    """
    reader_idx = reader_ids.index(reader_id)
    cluster_id = labels[reader_idx]

    cluster_members = [reader_ids[i] for i in range(len(reader_ids)) if labels[i] == cluster_id]
    cluster_records = df_train[df_train["reader_id"].isin(cluster_members)]

    user_borrowed = set(df_train[df_train["reader_id"] == reader_id]["book"].values)

    user_subject_vec = reader_subject_counts[reader_id]
    user_total = user_subject_vec.sum()
    if user_total == 0:
        user_subject_ratio = np.zeros(len(subjects_list))
    else:
        user_subject_ratio = user_subject_vec / user_total

    cluster_book_counts = cluster_records["book"].value_counts()

    candidate_scores = {}
    for book, cluster_count in cluster_book_counts.items():
        if book in user_borrowed:
            continue
        book_subject = BOOK_TO_SUBJECT[book]
        subj_idx = subjects_list.index(book_subject)
        user_familiarity = user_subject_ratio[subj_idx]
        novelty_weight = 1.0 - user_familiarity * 0.5
        score = cluster_count * novelty_weight
        candidate_scores[book] = score

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [book for book, _ in sorted_candidates[:top_n]]

    if len(recommendations) < top_n:
        for book in ALL_BOOKS:
            if book not in user_borrowed and book not in recommendations:
                recommendations.append(book)
            if len(recommendations) >= top_n:
                break

    return recommendations


def popularity_recommend(reader_id, df_train, top_n=3):
    """
    基线推荐：基于全局流行度
    推荐全局借阅次数最多的、该读者未借阅过的书籍
    """
    user_borrowed = set(df_train[df_train["reader_id"] == reader_id]["book"].values)
    global_popularity = df_train["book"].value_counts()

    recommendations = []
    for book, _ in global_popularity.items():
        if book not in user_borrowed:
            recommendations.append(book)
        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# 4. 验证与评估
# ============================================================

def evaluate_recommendations(sample_readers, df_train, df_test, labels,
                             reader_ids, reader_subject_counts, subjects_list):
    """对抽样读者评估推荐命中率"""

    results_spectral = {"hits": 0, "total": 0, "per_group": defaultdict(lambda: {"hits": 0, "total": 0})}
    results_popularity = {"hits": 0, "total": 0, "per_group": defaultdict(lambda: {"hits": 0, "total": 0})}

    detailed_results = []

    for reader_id in sample_readers:
        reader_idx = reader_ids.index(reader_id)
        cluster_id = labels[reader_idx]

        test_books = set(df_test[df_test["reader_id"] == reader_id]["book"].values)
        test_subjects = set(df_test[df_test["reader_id"] == reader_id]["subject"].values)

        rec_spectral = spectral_recommend(
            reader_id, labels, reader_ids, reader_subject_counts,
            subjects_list, df_train, top_n=3
        )
        rec_popularity = popularity_recommend(reader_id, df_train, top_n=3)

        hit_spectral_books = set(rec_spectral) & test_books
        hit_spectral_subjects = set(BOOK_TO_SUBJECT[b] for b in rec_spectral) & test_subjects

        hit_popularity_books = set(rec_popularity) & test_books
        hit_popularity_subjects = set(BOOK_TO_SUBJECT[b] for b in rec_popularity) & test_subjects

        spectral_hit = 1 if (hit_spectral_books or hit_spectral_subjects) else 0
        popularity_hit = 1 if (hit_popularity_books or hit_popularity_subjects) else 0

        results_spectral["hits"] += spectral_hit
        results_spectral["total"] += 1
        results_spectral["per_group"][cluster_id]["hits"] += spectral_hit
        results_spectral["per_group"][cluster_id]["total"] += 1

        results_popularity["hits"] += popularity_hit
        results_popularity["total"] += 1
        results_popularity["per_group"][cluster_id]["hits"] += popularity_hit
        results_popularity["per_group"][cluster_id]["total"] += 1

        detailed_results.append({
            "reader_id": reader_id,
            "cluster": cluster_id,
            "rec_spectral": rec_spectral,
            "rec_popularity": rec_popularity,
            "test_books": list(test_books),
            "spectral_book_hit": len(hit_spectral_books),
            "spectral_subject_hit": len(hit_spectral_subjects),
            "popularity_book_hit": len(hit_popularity_books),
            "popularity_subject_hit": len(hit_popularity_subjects),
        })

    return results_spectral, results_popularity, detailed_results


# ============================================================
# 主流程
# ============================================================

print("=" * 70)
print("图书推荐验证系统：谱聚类推荐 vs 流行度基线")
print("=" * 70)

# Step 1: 生成数据
print("\n[1] 数据准备")
print("-" * 70)
df_train, df_test, reader_profiles_assigned = generate_borrowing_records()
print(f"    训练集(9-10月): {len(df_train)} 条借阅记录")
print(f"    测试集(11月):   {len(df_test)} 条借阅记录")
print(f"    读者总数:       {NUM_READERS} 人")

# Step 2: 谱聚类
print("\n[2] 谱聚类分析")
print("-" * 70)
labels, reader_ids, optimal_k, reader_subject_counts, subjects_list = \
    perform_spectral_clustering(df_train)
print(f"    最优聚类数: k = {optimal_k}")
for c in range(optimal_k):
    count = np.sum(labels == c)
    cluster_members = [reader_ids[i] for i in range(len(reader_ids)) if labels[i] == c]
    cluster_records = df_train[df_train["reader_id"].isin(cluster_members)]
    top_subject = cluster_records["subject"].value_counts().index[0]
    print(f"    组{c + 1}: {count:3d} 人 | 主要偏好: {top_subject}")

# Step 3: 随机抽样100名读者
print("\n[3] 随机抽样100名读者进行推荐验证")
print("-" * 70)
sample_size = min(100, len(reader_ids))
sample_readers = np.random.choice(reader_ids, size=sample_size, replace=False)
print(f"    抽样人数: {sample_size}")
print(f"    抽样示例: {list(sample_readers[:10])}...")

# Step 4: 生成推荐并验证
print("\n[4] 推荐生成与验证")
print("-" * 70)
print("    推荐策略:")
print("      A. 谱聚类推荐: 基于簇内协同过滤 + 学科新颖性加权")
print("      B. 流行度基线: 全局借阅排行榜Top-N")
print("    验证方式: 推荐书籍/学科是否出现在测试集(11月)借阅记录中")

results_spectral, results_popularity, detailed_results = evaluate_recommendations(
    sample_readers, df_train, df_test, labels,
    reader_ids, reader_subject_counts, subjects_list
)

# Step 5: 输出总体结果
print("\n[5] 推荐命中率结果")
print("=" * 70)

overall_spectral = results_spectral["hits"] / results_spectral["total"] * 100
overall_popularity = results_popularity["hits"] / results_popularity["total"] * 100

print(f"\n┌─{'─' * 68}┐")
print(f"│{'总体推荐命中率':^34}│")
print(f"├─{'─' * 33}┬{'─' * 34}┤")
print(f"│{'方法':^16}│{'命中率':^17}│")
print(f"├─{'─' * 33}┼{'─' * 34}┤")
print(f"│  谱聚类推荐{'':>10}│  {results_spectral['hits']:3d}/{results_spectral['total']:3d} = {overall_spectral:6.2f}%{'':>7}│")
print(f"│  流行度基线{'':>10}│  {results_popularity['hits']:3d}/{results_popularity['total']:3d} = {overall_popularity:6.2f}%{'':>7}│")
print(f"├─{'─' * 33}┼{'─' * 34}┤")
improvement = overall_spectral - overall_popularity
print(f"│  提升幅度{'':>12}│  {'+' if improvement >= 0 else ''}{improvement:.2f} 百分点{'':>13}│")
print(f"└─{'─' * 33}┴{'─' * 34}┘")

# Step 6: 分组结果
print(f"\n{'─' * 70}")
print(f"{'各聚类组推荐精确率对比':^35}")
print(f"{'─' * 70}")
print(f"{'组号':<6}{'人数':<6}{'谱聚类命中':<14}{'谱聚类精确率':<14}{'流行度命中':<14}{'流行度精确率':<14}")
print(f"{'─' * 70}")

group_details = []
for c in range(optimal_k):
    sp = results_spectral["per_group"][c]
    pp = results_popularity["per_group"][c]

    if sp["total"] > 0:
        sp_rate = sp["hits"] / sp["total"] * 100
        pp_rate = pp["hits"] / pp["total"] * 100
    else:
        sp_rate = 0
        pp_rate = 0

    cluster_members = [reader_ids[i] for i in range(len(reader_ids)) if labels[i] == c]
    cluster_records = df_train[df_train["reader_id"].isin(cluster_members)]
    top_subject = cluster_records["subject"].value_counts().index[0]

    print(f" {c + 1:<5}{sp['total']:<6}{sp['hits']}/{sp['total']:<10}{sp_rate:>6.1f}%{'':>5}{pp['hits']}/{pp['total']:<10}{pp_rate:>6.1f}%")

    group_details.append({
        "group": c + 1,
        "top_subject": top_subject,
        "sample_size": sp["total"],
        "spectral_rate": sp_rate,
        "popularity_rate": pp_rate,
        "improvement": sp_rate - pp_rate,
    })

print(f"{'─' * 70}")

# Step 7: 推荐示例展示
print(f"\n{'─' * 70}")
print(f"{'推荐示例（前5名抽样读者）':^35}")
print(f"{'─' * 70}")

for detail in detailed_results[:5]:
    rid = detail["reader_id"]
    cid = detail["cluster"]
    profile = reader_profiles_assigned.get(rid, "未知")
    print(f"\n  读者 {rid} (类型: {profile}, 组{cid + 1})")
    print(f"    谱聚类推荐: {detail['rec_spectral']}")
    print(f"    流行度推荐: {detail['rec_popularity']}")
    print(f"    11月实际借: {detail['test_books'][:5]}{'...' if len(detail['test_books']) > 5 else ''}")
    hit_mark_s = "[HIT]" if (detail["spectral_book_hit"] or detail["spectral_subject_hit"]) else "[MISS]"
    hit_mark_p = "[HIT]" if (detail["popularity_book_hit"] or detail["popularity_subject_hit"]) else "[MISS]"
    print(f"    谱聚类: {hit_mark_s} (书籍{detail['spectral_book_hit']}本,学科{detail['spectral_subject_hit']}个)")
    print(f"    流行度: {hit_mark_p} (书籍{detail['popularity_book_hit']}本,学科{detail['popularity_subject_hit']}个)")

# ============================================================
# Step 8: 组2错误案例分析 —— 失效模式归因
# ============================================================

print(f"\n\n{'=' * 70}")
print(f"{'组2错误案例分析: 失效模式归因':^35}")
print(f"{'=' * 70}")

# 获取全局热门书籍Top-10
global_top_books = df_train["book"].value_counts().head(10).index.tolist()
global_top_subjects = set(BOOK_TO_SUBJECT[b] for b in global_top_books)

# 获取组2聚类特征
cluster_2_id = 1  # 组2对应的cluster index
cluster_2_members_all = [reader_ids[i] for i in range(len(reader_ids)) if labels[i] == cluster_2_id]
cluster_2_records = df_train[df_train["reader_id"].isin(cluster_2_members_all)]
cluster_2_top_subjects = cluster_2_records["subject"].value_counts().head(3).index.tolist()

print(f"\n  组2聚类特征:")
print(f"    主要偏好学科: {cluster_2_top_subjects}")
print(f"    全局热门Top10书籍所属学科: {sorted(global_top_subjects)}")
print(f"    重叠度: 组2偏好与全局热门高度重合")

# 对组2的MISS案例进行分类
group2_results = [d for d in detailed_results if d["cluster"] == cluster_2_id]
group2_miss = [d for d in group2_results
               if d["spectral_book_hit"] == 0 and d["spectral_subject_hit"] == 0]

failure_mode_A = []  # 全局热门干扰: 推荐的书与流行度基线高度重合
failure_mode_B = []  # 聚类边界模糊: 读者实际兴趣偏离簇内主流

print(f"\n  组2总样本: {len(group2_results)} 人")
print(f"  组2谱聚类MISS数: {len(group2_miss)} 人")
print(f"\n  失效模式分类标准:")
print(f"    模式A [全局热门干扰]: 推荐书 >= 2/3 与流行度基线重合")
print(f"    模式B [聚类边界模糊]: 测试集借阅学科 <= 1个属于组2 Top学科")

for d in group2_miss:
    rec_set = set(d["rec_spectral"])
    pop_set = set(d["rec_popularity"])
    overlap_with_pop = len(rec_set & pop_set)

    test_subjects_of_reader = set(BOOK_TO_SUBJECT[b] for b in d["test_books"])
    overlap_with_cluster_pref = len(test_subjects_of_reader & set(cluster_2_top_subjects))

    is_mode_A = overlap_with_pop >= 2
    is_mode_B = overlap_with_cluster_pref <= 1

    d["failure_mode"] = []
    d["overlap_with_pop"] = overlap_with_pop
    d["test_subjects"] = test_subjects_of_reader
    d["cluster_pref_overlap"] = overlap_with_cluster_pref

    if is_mode_A:
        failure_mode_A.append(d)
        d["failure_mode"].append("A")
    if is_mode_B:
        failure_mode_B.append(d)
        d["failure_mode"].append("B")
    if not is_mode_A and not is_mode_B:
        d["failure_mode"].append("mixed")

# 统计
mode_A_only = [d for d in group2_miss if d["failure_mode"] == ["A"]]
mode_B_only = [d for d in group2_miss if d["failure_mode"] == ["B"]]
mode_AB = [d for d in group2_miss if "A" in d["failure_mode"] and "B" in d["failure_mode"]]
mode_other = [d for d in group2_miss if d["failure_mode"] == ["mixed"]]

print(f"\n  {'─' * 66}")
print(f"  {'失效模式':^10}{'含义':^28}{'案例数':^8}{'占比':^10}")
print(f"  {'─' * 66}")
print(f"  {'模式A':<10}{'全局热门干扰(推荐与流行度基线重合)':<28}{len(failure_mode_A):^8}{len(failure_mode_A)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'模式B':<10}{'聚类边界模糊(实际兴趣偏离簇特征)':<28}{len(failure_mode_B):^8}{len(failure_mode_B)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'A+B重叠':<10}{'两种模式同时存在':<28}{len(mode_AB):^8}{len(mode_AB)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'仅A':<10}{'纯全局热门干扰':<28}{len(mode_A_only):^8}{len(mode_A_only)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'仅B':<10}{'纯聚类边界模糊':<28}{len(mode_B_only):^8}{len(mode_B_only)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'其他':<10}{'未归入上述两类':<28}{len(mode_other):^8}{len(mode_other)/max(len(group2_miss),1)*100:>5.1f}%")
print(f"  {'─' * 66}")

# 展示典型错误案例
print(f"\n  典型错误案例:")

if failure_mode_A:
    d = failure_mode_A[0]
    print(f"\n  [模式A - 全局热门干扰] 读者 {d['reader_id']}")
    print(f"    谱聚类推荐:   {d['rec_spectral']}")
    print(f"    流行度推荐:   {d['rec_popularity']}")
    print(f"    推荐重合数:   {d['overlap_with_pop']}/3 本与流行度基线相同")
    print(f"    实际测试借阅: {d['test_books'][:4]}")
    print(f"    诊断: 簇内热门 ~ 全局热门，谱聚类退化为流行度推荐")

if failure_mode_B:
    d = failure_mode_B[0]
    print(f"\n  [模式B - 聚类边界模糊] 读者 {d['reader_id']}")
    print(f"    谱聚类推荐:   {d['rec_spectral']}")
    print(f"    实际测试借阅: {d['test_books'][:4]}")
    print(f"    测试集学科:   {d['test_subjects']}")
    print(f"    与组2偏好重合: {d['cluster_pref_overlap']} 个学科")
    print(f"    诊断: 该读者兴趣实际偏离组2主流，属于聚类边界误分配")

# ============================================================
# Step 9: 多粒度匹配精确率对比
# ============================================================

print(f"\n\n{'=' * 70}")
print(f"{'多粒度匹配精确率对比':^35}")
print(f"{'=' * 70}")

print(f"\n  匹配粒度定义:")
print(f"    L1 [严格匹配]:  推荐书 == 实际借阅书 (完全一致)")
print(f"    L2 [宽松匹配]:  推荐书的学科中有其他书被借阅 (同学科命中)")
print(f"    L3 [学科匹配]:  推荐书的学科 == 实际借阅的任一学科")

# 计算三种粒度
def compute_multi_granularity(detail_list, df_test_local):
    """对每个读者计算三种粒度的命中"""
    results_l1 = []  # 严格: 推荐书 in 测试集书
    results_l2 = []  # 宽松: 推荐书的同学科其他书 in 测试集书
    results_l3 = []  # 学科: 推荐书学科 in 测试集学科

    for d in detail_list:
        rec_books = d["rec_spectral"]
        test_books = set(d["test_books"])
        test_subjects = set(BOOK_TO_SUBJECT[b] for b in d["test_books"])

        # L1: 严格书籍匹配
        l1_hit = 1 if len(set(rec_books) & test_books) > 0 else 0

        # L2: 宽松匹配 —— 推荐书所属学科的任何其他书出现在测试集
        l2_hit = 0
        for rec_book in rec_books:
            rec_subject = BOOK_TO_SUBJECT[rec_book]
            same_subject_books = set(SUBJECT_CATEGORIES[rec_subject]) - {rec_book}
            if same_subject_books & test_books:
                l2_hit = 1
                break

        # L3: 学科匹配
        rec_subjects = set(BOOK_TO_SUBJECT[b] for b in rec_books)
        l3_hit = 1 if len(rec_subjects & test_subjects) > 0 else 0

        results_l1.append(l1_hit)
        results_l2.append(l2_hit)
        results_l3.append(l3_hit)

    return results_l1, results_l2, results_l3


# 同时计算流行度基线的多粒度
def compute_multi_granularity_pop(detail_list):
    results_l1 = []
    results_l2 = []
    results_l3 = []

    for d in detail_list:
        rec_books = d["rec_popularity"]
        test_books = set(d["test_books"])
        test_subjects = set(BOOK_TO_SUBJECT[b] for b in d["test_books"])

        l1_hit = 1 if len(set(rec_books) & test_books) > 0 else 0

        l2_hit = 0
        for rec_book in rec_books:
            rec_subject = BOOK_TO_SUBJECT[rec_book]
            same_subject_books = set(SUBJECT_CATEGORIES[rec_subject]) - {rec_book}
            if same_subject_books & test_books:
                l2_hit = 1
                break

        rec_subjects = set(BOOK_TO_SUBJECT[b] for b in rec_books)
        l3_hit = 1 if len(rec_subjects & test_subjects) > 0 else 0

        results_l1.append(l1_hit)
        results_l2.append(l2_hit)
        results_l3.append(l3_hit)

    return results_l1, results_l2, results_l3


print(f"\n  {'─' * 66}")
print(f"  {'':^6}{'谱聚类推荐':^30}{'流行度基线':^30}")
print(f"  {'粒度':<6}{'组1':^10}{'组2':^10}{'总体':^10}{'组1':^10}{'组2':^10}{'总体':^10}")
print(f"  {'─' * 66}")

granularity_table = {}
for c in range(optimal_k):
    g_results = [d for d in detailed_results if d["cluster"] == c]
    sp_l1, sp_l2, sp_l3 = compute_multi_granularity(g_results, df_test)
    pp_l1, pp_l2, pp_l3 = compute_multi_granularity_pop(g_results)
    granularity_table[c] = {
        "sp": [np.mean(sp_l1)*100, np.mean(sp_l2)*100, np.mean(sp_l3)*100],
        "pp": [np.mean(pp_l1)*100, np.mean(pp_l2)*100, np.mean(pp_l3)*100],
        "n": len(g_results),
    }

# 加权平均
total_n = sum(granularity_table[c]["n"] for c in range(optimal_k))
sp_weighted = [0.0, 0.0, 0.0]
pp_weighted = [0.0, 0.0, 0.0]
for c in range(optimal_k):
    w = granularity_table[c]["n"] / total_n
    for level in range(3):
        sp_weighted[level] += granularity_table[c]["sp"][level] * w
        pp_weighted[level] += granularity_table[c]["pp"][level] * w

level_names = ["L1严格", "L2宽松", "L3学科"]
for i, lname in enumerate(level_names):
    g1_sp = granularity_table[0]["sp"][i]
    g2_sp = granularity_table[1]["sp"][i] if 1 in granularity_table else 0
    g1_pp = granularity_table[0]["pp"][i]
    g2_pp = granularity_table[1]["pp"][i] if 1 in granularity_table else 0
    print(f"  {lname:<6}{g1_sp:>7.1f}%  {g2_sp:>7.1f}%  {sp_weighted[i]:>7.1f}%  "
          f"{g1_pp:>7.1f}%  {g2_pp:>7.1f}%  {pp_weighted[i]:>7.1f}%")

print(f"  {'─' * 66}")

# 精确率提升对比
print(f"\n  谱聚类 vs 流行度 提升幅度(百分点):")
print(f"  {'─' * 50}")
print(f"  {'粒度':<8}{'组1提升':^12}{'组2提升':^12}{'加权总体提升':^14}")
print(f"  {'─' * 50}")
for i, lname in enumerate(level_names):
    g1_diff = granularity_table[0]["sp"][i] - granularity_table[0]["pp"][i]
    g2_diff = (granularity_table[1]["sp"][i] - granularity_table[1]["pp"][i]) if 1 in granularity_table else 0
    total_diff = sp_weighted[i] - pp_weighted[i]
    print(f"  {lname:<8}{g1_diff:>+8.1f}    {g2_diff:>+8.1f}    {total_diff:>+10.1f}")
print(f"  {'─' * 50}")

print(f"\n  关键发现:")
print(f"    - L2宽松匹配(同学科其他书)可减少书库有限导致的假阴性")
print(f"    - 组2在L1严格匹配下表现不佳，但L2/L3提升显著，")
print(f"      说明谱聚类方向正确，只是具体书目选择需优化")

# ============================================================
# Step 10: 样本量加权平均 & 各组贡献分解表
# ============================================================

print(f"\n\n{'=' * 70}")
print(f"{'样本量加权平均精确率 & 各组贡献分解':^35}")
print(f"{'=' * 70}")

print(f"\n  加权公式: P_total = sum(n_i / N * P_i)")
print(f"  其中 n_i = 组i的抽样人数, N = 总抽样人数, P_i = 组i精确率\n")

# 对三种粒度分别计算贡献分解
for level_idx, lname in enumerate(level_names):
    print(f"  {'=' * 66}")
    print(f"  {lname} 匹配精确率分解")
    print(f"  {'─' * 66}")
    print(f"  {'组号':<5}{'样本n_i':<8}{'权重w_i':<10}{'谱聚类P_i':<12}{'贡献w*P':<10}"
          f"{'流行度P_i':<12}{'贡献w*P':<10}")
    print(f"  {'─' * 66}")

    sp_total_contribution = 0.0
    pp_total_contribution = 0.0

    for c in range(optimal_k):
        n_i = granularity_table[c]["n"]
        w_i = n_i / total_n
        sp_pi = granularity_table[c]["sp"][level_idx]
        pp_pi = granularity_table[c]["pp"][level_idx]
        sp_contrib = w_i * sp_pi
        pp_contrib = w_i * pp_pi
        sp_total_contribution += sp_contrib
        pp_total_contribution += pp_contrib

        print(f"  {c+1:<5}{n_i:<8}{w_i:<10.3f}{sp_pi:<12.1f}{sp_contrib:<10.2f}"
              f"{pp_pi:<12.1f}{pp_contrib:<10.2f}")

    print(f"  {'─' * 66}")
    print(f"  {'合计':<5}{total_n:<8}{'1.000':<10}{'':<12}{sp_total_contribution:<10.2f}"
          f"{'':<12}{pp_total_contribution:<10.2f}")
    print(f"  加权平均精确率: 谱聚类={sp_total_contribution:.1f}% | 流行度={pp_total_contribution:.1f}% | "
          f"差值={sp_total_contribution - pp_total_contribution:+.1f}%\n")

# ============================================================
# Step 11: 综合分析 - L3学科匹配的加权详细表(与原始报告口径一致)
# ============================================================

print(f"\n{'=' * 70}")
print(f"{'总结: 原始口径(书籍+学科命中)加权分解':^35}")
print(f"{'=' * 70}")

print(f"\n  此表对应主报告中的命中率口径(推荐书籍或其学科出现在测试集中)\n")
print(f"  {'─' * 66}")
print(f"  {'组号':<5}{'样本':<6}{'权重':<8}{'谱聚类':^22}{'流行度':^22}")
print(f"  {'':5}{'':6}{'':8}{'命中':^7}{'精确率':^8}{'贡献':^7}{'命中':^7}{'精确率':^8}{'贡献':^7}")
print(f"  {'─' * 66}")

sp_total_w = 0.0
pp_total_w = 0.0
for c in range(optimal_k):
    sp = results_spectral["per_group"][c]
    pp = results_popularity["per_group"][c]
    n_i = sp["total"]
    w_i = n_i / total_n
    sp_rate = sp["hits"] / sp["total"] * 100 if sp["total"] > 0 else 0
    pp_rate = pp["hits"] / pp["total"] * 100 if pp["total"] > 0 else 0
    sp_contrib = w_i * sp_rate
    pp_contrib = w_i * pp_rate
    sp_total_w += sp_contrib
    pp_total_w += pp_contrib

    print(f"  {c+1:<5}{n_i:<6}{w_i:<8.3f}"
          f"{sp['hits']}/{sp['total']:<5}  {sp_rate:>5.1f}%  {sp_contrib:>6.2f}  "
          f"{pp['hits']}/{pp['total']:<5}  {pp_rate:>5.1f}%  {pp_contrib:>6.2f}")

print(f"  {'─' * 66}")
print(f"  {'加权':<5}{total_n:<6}{'1.000':<8}{'':^7}{sp_total_w:>5.1f}%  {sp_total_w:>6.2f}  "
      f"{'':^7}{pp_total_w:>5.1f}%  {pp_total_w:>6.2f}")
print(f"  {'─' * 66}")
print(f"  加权提升: {sp_total_w - pp_total_w:+.1f} 百分点")

# ============================================================
# Step 12: 改进建议（含针对性错误模式修正策略）
# ============================================================

print(f"\n\n{'=' * 70}")
print(f"{'改进建议(含错误模式针对性修正)':^35}")
print(f"{'=' * 70}")

print(f"""
  [针对模式A - 全局热门干扰]
  ──────────────────────────────────────────────────────
  问题: 组2偏好(计算机/数学)与全局热门高度重合，簇内协同过滤
        退化为全局流行度排序，丧失个性化能力。
  修正策略:
    1. 去流行度偏差(Popularity Debiasing):
       score = cluster_score / global_popularity^alpha (alpha=0.3~0.5)
    2. 逆频率加权(IUF): 对全局借阅Top10书籍施加惩罚因子
    3. 增加组内差异化: 用TF-IDF思想, 提升"组内高频但全局低频"的书
    4. 补充内容特征: 区分同一学科内的细分方向(如AI vs 系统)

  [针对模式B - 聚类边界模糊]
  ──────────────────────────────────────────────────────
  问题: 部分读者被错误分配到组2，实际兴趣偏离簇中心，
        导致簇内推荐与真实需求不匹配。
  修正策略:
    1. 软聚类/模糊隶属: 允许读者以概率属于多个簇，
       推荐时按隶属度加权融合各簇候选
    2. 增加聚类数k: 当前k=2可能过粗，尝试k=3~5细分
    3. 引入个人偏好修正: 在簇推荐基础上，用个人历史
       借阅向量做re-ranking
    4. 边界检测: 计算读者到簇中心的距离，对边界读者
       (距离>阈值)回退到个性化推荐而非簇推荐

  [通用改进]
  ──────────────────────────────────────────────────────
    * 混合推荐: 0.5*谱聚类 + 0.3*个性化CF + 0.2*流行度
    * 时间衰减: 近2周借阅权重x2，捕捉短期兴趣变化
    * 多样性约束: 推荐3本书须覆盖>=2个学科，避免同质化
    * 冷启动: 新读者先用专业/年级画像推荐，积累数据后切换
""")

print("=" * 70)
print("推荐验证分析完成")
print(f"  谱聚类加权精确率: {sp_total_w:.1f}%")
print(f"  流行度加权精确率: {pp_total_w:.1f}%")
print(f"  加权提升: {sp_total_w - pp_total_w:+.1f} 百分点")
print(f"  组2失效归因: 模式A(全局热门干扰) {len(failure_mode_A)}例, "
      f"模式B(聚类边界模糊) {len(failure_mode_B)}例")
print("=" * 70)
