"""Figures for the group-meeting summary (static PNG, light surface, palette validated with the dataviz validator)."""
import os, pandas as pd, matplotlib
FIG = os.environ.get('FIG_DIR', 'figures'); os.makedirs(FIG, exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = [f for f in font_manager.findSystemFonts() if 'DroidSansFallback' in f]
if fp: font_manager.fontManager.addfont(fp[0]); plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=fp[0]).get_name()]
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C = {'统计基准': '#2a78d6', 'AI 实名版': '#eb6834', 'AI 匿名 / 训练截止后': '#1baf7a'}
def bars(items, title, xlabel, path, note):
    items = sorted(items, key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.6), dpi=160); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    y = range(len(items))
    for i, (lab, v, grp) in enumerate(items):
        ax.barh(i, v, height=0.62, color=C[grp], edgecolor=SURF, linewidth=2)
        ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=9, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([x[0] for x in items], fontsize=9.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2); ax.tick_params(axis='x', colors=INK2, labelsize=8.5); ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.15)
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=28)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in C.items() if any(x[2] == g for x in items)]
    ax.legend(hs, [g for g in C if any(x[2] == g for x in items)], loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, handlelength=1.2)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, facecolor=SURF); plt.close(fig)
M = pd.read_csv('data/metrics_v5.csv'); A = M[(M.split == 'test 2014-2020') & (M.h == 'all')].set_index('method').mem_MAE
cmpA = pd.read_csv('data/compare_LLM_anon_2014-2020.csv', index_col=0); cmpN = pd.read_csv('data/compare_LLM_named_2014-2020.csv', index_col=0)
lab = {'B14_combo_plus_text_h0': '统计组合 B14（最强）', 'B0_prior_own': '沿用本人上次的点 B0', 'B5_prior_plus_policy_change': '上次 + 期间利率变动 B5',
       'B8_ar1_own': '时间序列 AR(1) B8', 'B7_xgb_revision': 'XGBoost B7', 'B1a_prev_median': '上次委员会中位数 B1a', 'B2c_rt_rule_smooth': '实时反应函数 B2c',
       'B9_market_forward': '国债市场预期 B9', 'B2a_taylor1993': '泰勒规则（1993）B2a'}
items = [(lab[k], A[k], '统计基准') for k in lab] + [('AI 实名版（gpt-4.1-mini）', cmpN.loc['LLM_named', 'MAE'], 'AI 实名版'), ('AI 匿名版（同样数字，去掉人名日期）', cmpA.loc['LLM_anon', 'MAE'], 'AI 匿名 / 训练截止后')]
bars(items, '个人点阵图预测误差（2014–2020 年测试期，越短越好）', '平均绝对误差 MAE（百分点）', FIG + '/fig1_individual_2014_2020.png',
     '27 次 SEP、2,013 个点；所有方法只用会前信息，需估参数的模型用扩展窗口。')
R = pd.read_csv('data/post_cutoff_results.csv'); B = R.groupby('method').median_err.mean()
lab2 = {'S1 previous distribution': ('沿用上次的点阵图', '统计基准'), 'S3 previous + policy change': ('上次 + 期间利率变动', '统计基准'), 'S2 no change': ('利率不变', '统计基准'),
        'S4 market forward': ('国债市场预期', '统计基准'), 'L1 LLM (gpt-4.1-mini)': ('AI 只给数字', 'AI 匿名 / 训练截止后'), 'L2 LLM + materials': ('AI 数字 + 会前材料', 'AI 匿名 / 训练截止后'),
        'L3 LLM + materials, anchored': ('AI 以上次为起点 + 材料', 'AI 匿名 / 训练截止后'), 'L4 LLM anchored, no materials': ('AI 以上次为起点', 'AI 匿名 / 训练截止后')}
bars([(v[0], B[k], v[1]) for k, v in lab2.items()], '委员会中位数预测误差（模型训练截止后的 9 次 SEP）', '平均绝对误差 MAE（百分点）',
     FIG + '/fig2_post_cutoff.png', '2024-09 至 2026-09，41 个“会议 × 目标年份”；gpt-4.1-mini 训练数据截至 2024-06，不可能记得这些会议。')
print('ok')
