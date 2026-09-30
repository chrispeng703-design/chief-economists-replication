# -*- coding: utf-8 -*-
"""
官方话语扩散与词典情感效度 —— 预先设定的分析
Official discourse diffusion and the validity of dictionary sentiment (CCEF corpus)

原则：下面 A1–A5 五组分析在看到结果之前就固定下来，无论显著与否全部写进论文。
不要根据结果增删分析或换时间节点，否则就是 p-hacking。

Windows 运行：
    pip install pandas openpyxl jieba pyfixest
    python discourse_analysis.py
"""
import re
import numpy as np
import pandas as pd
import jieba
import pyfixest as pf

# ======================= 配置：按你的文件修改 =======================
ARTICLE_FILE = r"D:\ccef\main_sample.xlsx"   # 主样本文章级文件（2021-01 至 2026-08，9,291 篇）
COL_ID, COL_AUTHOR, COL_DATE, COL_TEXT, COL_GROUP = "article_id", "author", "date", "text", "group"
POS_FILE = r"D:\ccef\dict\yao_pos.txt"      # Yao et al. (2021) 正面词，每行一个（从 step1 用的 cntext 词典导出）
NEG_FILE = r"D:\ccef\dict\yao_neg.txt"      # Yao et al. (2021) 负面词
LLM_FILE = r"D:\ccef\llm_codes.csv"         # 可选：含 COL_ID 与 stance 列的 LLM 编码；没有就改成 None
COL_STANCE = "stance"
OUT_FILE = r"D:\ccef\discourse_results.xlsx"
# ====================================================================

CEWC_DATE = pd.Timestamp("2023-12-12")      # 中央经济工作会议提出"唱响中国经济光明论"
NOTICE_DATE = pd.Timestamp("2024-12-18")    # 中证协通知（这里只作时期分界，不做 DiD）

OFFICIAL = ["高质量发展", "新质生产力", "中国式现代化", "新发展格局", "新发展理念", "稳中求进",
            "稳中向好", "回升向好", "长期向好", "光明论", "以进促稳", "先立后破", "统筹发展和安全",
            "共同富裕", "习近平", "总书记", "党中央", "韧性强", "潜力大", "活力足"]
LEADER = {"习近平", "总书记", "党中央"}
HAN = re.compile(r"[\u4e00-\u9fff]")
SENT_SPLIT = re.compile(r"[。！？；!?;\n]+")


def load_words(path):
    with open(path, encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def tone(p, n):
    return (p - n) / (p + n) if (p + n) > 0 else np.nan


def score_article(text, pos, neg, official):
    """返回：全文语调、剔除官方套话句后的语调、套话句语调、套话句占比、官方词/千词、领导人词/千词"""
    p_all = n_all = p_mask = n_mask = p_form = n_form = 0
    n_tok = n_off = n_lead = n_sent = n_form_sent = 0
    for sent in SENT_SPLIT.split(str(text)):
        toks = [t for t in jieba.lcut(sent, HMM=False) if HAN.search(t)]
        if not toks:
            continue
        n_sent += 1
        n_tok += len(toks)
        p = sum(t in pos for t in toks)
        n = sum(t in neg for t in toks)
        off = sum(t in official for t in toks)
        n_off += off
        n_lead += sum(t in LEADER for t in toks)
        p_all += p
        n_all += n
        if off > 0:                       # 含官方词的句子 = 套话句
            n_form_sent += 1
            p_form += p
            n_form += n
        else:
            p_mask += p
            n_mask += n
    k = 1000 / n_tok if n_tok else np.nan
    return dict(n_tok=n_tok, tone_all=tone(p_all, n_all), tone_masked=tone(p_mask, n_mask),
                tone_formulaic=tone(p_form, n_form),
                formulaic_share=n_form_sent / n_sent if n_sent else np.nan,
                official_per1k=n_off * k, leader_per1k=n_lead * k)


def fe(formula, data, label):
    fit = pf.feols(formula, data=data, vcov={"CRV1": COL_AUTHOR})
    out = fit.tidy().reset_index()
    out.insert(0, "model", label)
    out["N"] = fit._N
    return out


def main():
    pos, neg = load_words(POS_FILE), load_words(NEG_FILE)
    official = set(OFFICIAL)
    for w in pos | neg | official:
        jieba.add_word(w)

    df = pd.read_excel(ARTICLE_FILE) if ARTICLE_FILE.endswith("xlsx") else pd.read_csv(ARTICLE_FILE)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    scores = pd.DataFrame([score_article(t, pos, neg, official) for t in df[COL_TEXT]])
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df = df[df["n_tok"] >= 200].copy()
    df["loglen"] = np.log(df["n_tok"])
    df["P1"] = ((df[COL_DATE] >= CEWC_DATE) & (df[COL_DATE] < NOTICE_DATE)).astype(int)
    df["P2"] = (df[COL_DATE] >= NOTICE_DATE).astype(int)
    df["half"] = df[COL_DATE].dt.year.astype(str) + "H" + np.where(df[COL_DATE].dt.month <= 6, "1", "2")
    res = []

    # A1 官方话语扩散：经济学家固定效应下的前后比较（描述性，不作因果解释）
    for y in ["official_per1k", "leader_per1k", "formulaic_share"]:
        res.append(fe(f"{y} ~ P1 + P2 + loglen | {COL_AUTHOR}", df, f"A1 {y}"))

    # A2 按机构类型分别估计扩散（异质性，仅描述）
    for g, sub in df.groupby(COL_GROUP):
        if sub[COL_AUTHOR].nunique() >= 5:
            res.append(fe(f"official_per1k ~ P1 + P2 + loglen | {COL_AUTHOR}", sub, f"A2 official [{g}]"))

    # A3 词典语调是否被套话污染：全文语调 vs 剔除套话句语调，同一篇文章堆叠比较
    both = df.dropna(subset=["tone_all", "tone_masked"])
    long = both.melt(id_vars=[COL_ID, COL_AUTHOR, "P1", "P2", "loglen"],
                     value_vars=["tone_all", "tone_masked"], var_name="measure", value_name="tone")
    long["masked"] = (long["measure"] == "tone_masked").astype(int)
    long["P1_x_masked"], long["P2_x_masked"] = long["P1"] * long["masked"], long["P2"] * long["masked"]
    for y in ["tone_all", "tone_masked"]:
        res.append(fe(f"{y} ~ P1 + P2 + loglen | {COL_AUTHOR}", both, f"A3 {y}"))
    res.append(fe(f"tone ~ P1 + P2 + masked + P1_x_masked + P2_x_masked + loglen | {COL_AUTHOR}",
                  long, "A3 gap test (Px_masked < 0 = 套话抬高了语调变化)"))
    # 套话句本身是否比其他句子更正面（文章内配对）
    pair = df.dropna(subset=["tone_formulaic", "tone_masked"])
    diff = pair["tone_formulaic"] - pair["tone_masked"]
    desc_pair = pd.DataFrame({"model": ["A3 套话句语调 − 其他句语调（文章内）"],
                              "mean": [diff.mean()], "se": [diff.std() / np.sqrt(len(diff))], "N": [len(diff)]})

    # A4 词典语调与 LLM 立场的背离（只用模型编码；没有人工验证时论文必须如实说明）
    if LLM_FILE:
        llm = pd.read_csv(LLM_FILE)[[COL_ID, COL_STANCE]]
        m = df.merge(llm, on=COL_ID).dropna(subset=[COL_STANCE, "tone_all", "tone_masked"])
        for c in ["tone_all", "tone_masked", COL_STANCE]:
            m["z_" + c] = (m[c] - m[c].mean()) / m[c].std()
        m["gap_all"] = m["z_tone_all"] - m["z_" + COL_STANCE]
        m["gap_masked"] = m["z_tone_masked"] - m["z_" + COL_STANCE]
        for y in ["z_" + COL_STANCE, "gap_all", "gap_masked"]:
            res.append(fe(f"{y} ~ P1 + P2 + loglen | {COL_AUTHOR}", m, f"A4 {y}"))
        rho = m[["tone_all", "tone_masked", COL_STANCE]].corr(method="spearman")

    # A5 描述性半年均值（作图用）
    desc = df.groupby("half")[["official_per1k", "leader_per1k", "formulaic_share",
                               "tone_all", "tone_masked", "tone_formulaic"]].mean()
    desc["articles"] = df.groupby("half").size()

    with pd.ExcelWriter(OUT_FILE) as w:
        pd.concat(res, ignore_index=True).to_excel(w, sheet_name="regressions", index=False)
        desc_pair.to_excel(w, sheet_name="formulaic_vs_other", index=False)
        desc.to_excel(w, sheet_name="half_year_means")
        if LLM_FILE:
            rho.to_excel(w, sheet_name="spearman")
        df.drop(columns=[COL_TEXT]).to_excel(w, sheet_name="article_scores", index=False)
    print(pd.concat(res, ignore_index=True).to_string())
    print("\n结果已保存：", OUT_FILE)


if __name__ == "__main__":
    main()
