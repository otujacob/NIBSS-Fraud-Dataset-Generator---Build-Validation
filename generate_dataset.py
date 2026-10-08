"""
NIBSS-calibrated synthetic inter-bank fraud dataset generator.

Fully vectorised; every behavioural feature is computed from the final
transaction amounts using only information available up to (and including)
the current transaction of the same customer (strictly causal).

Parameters that are taken from the NIBSS 2023 Annual Fraud Landscape are marked
[NIBSS]; parameters that are modelling assumptions without a published source
are marked [ASSUMED] and are documented as such in the accompanying paper.

Usage:  python generate_dataset.py --customers 25000 --prevalence 0.001 --out data
"""
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CHANNELS = ['ATM', 'ECOM', 'IB', 'Mobile', 'POS', 'Web']
BANKS = [f'Bank_{i}' for i in range(1, 11)]
LOCATIONS = ['Lagos', 'Abuja', 'Ogun', 'Oyo', 'Rivers', 'Other']
LOC_P = [0.28, 0.20, 0.12, 0.14, 0.11, 0.15]                       # [ASSUMED]
AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
AGE_P = [0.18, 0.30, 0.24, 0.15, 0.08, 0.05]                       # [ASSUMED]
MERCHANTS = ['Medical', 'ATM Withdrawal', 'Retail', 'Grocery', 'Education',
             'Transfer', 'Electronics', 'Transport', 'Airtime', 'Fashion',
             'Restaurant', 'Bill Payment', 'Entertainment']
MERCHANT_FRAUD_FACTOR = {                                           # [ASSUMED]
    'Medical': 0.3, 'ATM Withdrawal': 0.4, 'Retail': 0.7, 'Grocery': 0.5,
    'Education': 0.4, 'Transfer': 1.4, 'Electronics': 1.3, 'Transport': 0.9,
    'Airtime': 1.1, 'Fashion': 1.0, 'Restaurant': 0.8, 'Bill Payment': 1.2,
    'Entertainment': 1.1}
TECHNIQUES = ['SOCIAL_ENGINEERING', 'ROBBERY', 'CARD_THEFT', 'PIN_COMPROMISE',
              'PHISHING', 'OTHER']

# [NIBSS] 2023 fraud counts and losses (NGN million) by channel
NIBSS_CH_COUNT = {'Mobile': 47526, 'Web': 21887, 'POS': 17560,
                  'IB': 5381, 'ECOM': 2449, 'ATM': 722}
NIBSS_CH_LOSS_M = {'Mobile': 5691.7, 'Web': 2223.6, 'POS': 4415.8,
                   'IB': 4097.3, 'ECOM': 144.1, 'ATM': 63.6}
# [NIBSS] 2023 fraud counts by calendar month
NIBSS_MONTH_COUNT = [9234, 9492, 8829, 8234, 11716, 6240, 8649, 8207, 7503,
                     6860, 6367, 4289]
# [NIBSS] 2023 fraud counts by technique ("OTHER" = all remaining categories)
NIBSS_TECH_COUNT = {'SOCIAL_ENGINEERING': 62901, 'ROBBERY': 10179,
                    'CARD_THEFT': 6825, 'PIN_COMPROMISE': 5273,
                    'PHISHING': 4457, 'OTHER': 5985}

CH_P = np.array([0.05, 0.11, 0.11, 0.42, 0.16, 0.15])             # [ASSUMED] legit channel mix
CH_LEGIT_SHIFT = {'ATM': -0.30, 'ECOM': -0.10, 'IB': 0.50,         # [ASSUMED] log-amount shift
                  'Mobile': 0.0, 'POS': -0.10, 'Web': 0.10}
HOUR_W = np.array([.4, .3, .3, .3, .4, .5, .8, 1.2, 1.6, 2.0, 2.0, 2.0,
                   2.2, 2.2, 2.2, 2.0, 1.9, 2.1, 2.2, 2.0, 1.6, 1.2, .8, .5])  # [ASSUMED]
DOW_W = np.array([1.0, 1.0, 1.0, 1.05, 1.25, 1.1, 0.8])           # [ASSUMED] Mon..Sun
FRAUD_SIGMA = 0.9                                                  # [ASSUMED]
BURST_PROB = 0.25      # [ASSUMED] prob. a legit tx follows the previous one within minutes
BURST_MEAN_S = 900.0   # [ASSUMED]
FRAUD_FOLLOW_P = 0.40  # [ASSUMED] prob. that a fraud event is followed by another (takeover bursts)
WARM_DAYS = 90         # burn-in simulated before 1 Jan 2023 and discarded, so that history features
                       # are on comparable footing in every part of the released year
TOTAL_S = (365 + WARM_DAYS) * 86400
WARM_S = WARM_DAYS * 86400
KEY = 1e9              # customer key spacing (> seconds in a year)


ORIGIN = pd.Timestamp('2023-01-01') - pd.Timedelta(days=WARM_DAYS)   # Monday 3 Oct 2022


def month_of(sec):
    return (pd.to_datetime(sec, unit='s', origin=ORIGIN).month).values


def generate(n_customers=25000, prevalence=0.001, seed=42, drop_burnin=True):
    rng = np.random.default_rng(seed)

    # ---- customers ---------------------------------------------------------
    cust = pd.DataFrame({
        'customer_id': [f'CUST{i:07d}' for i in range(n_customers)],
        'age_group': rng.choice(AGE_GROUPS, n_customers, p=AGE_P),
        'location': rng.choice(LOCATIONS, n_customers, p=LOC_P),
        'bank': rng.choice(BANKS, n_customers),
        'pref': rng.choice(CHANNELS, n_customers, p=CH_P),
        'mu': rng.normal(10.8, 0.6, n_customers),         # customer spend level (log-NGN)
        'sig': rng.uniform(0.4, 0.9, n_customers),        # customer within-spend dispersion
        'n': np.clip(rng.lognormal(4.2, 0.7, n_customers).astype(int), 5, 1000)
             * (365 + WARM_DAYS) // 365,
    })
    cid = np.repeat(np.arange(n_customers), cust['n'].values)
    N = len(cid)

    # ---- legitimate timestamps: diurnal + weekly profile, bursty -----------
    nd = 365 + WARM_DAYS
    day_w = DOW_W[np.arange(nd) % 7]                   # day 0 = ORIGIN, a Monday
    day = rng.choice(nd, N, p=day_w / day_w.sum())
    hour = rng.choice(24, N, p=HOUR_W / HOUR_W.sum())
    t = day * 86400.0 + hour * 3600.0 + rng.uniform(0, 3600, N)
    order = np.lexsort((t, cid)); cid, t = cid[order], t[order]
    first = np.r_[True, cid[1:] != cid[:-1]]
    leader = first | (rng.random(N) > BURST_PROB)
    cl = np.cumsum(leader) - 1
    off = np.where(leader, 0.0, rng.exponential(BURST_MEAN_S, N))
    cs = np.cumsum(off)
    lead_idx = np.flatnonzero(leader)
    off = cs - cs[lead_idx][cl] + off * 0              # cumulative offset within cluster
    t = np.minimum(t[lead_idx][cl] + off, TOTAL_S - 1)

    # ---- legit channel / merchant / amount ---------------------------------
    pref = cust['pref'].values[cid]
    ch = np.where(rng.random(N) < 0.70, pref, rng.choice(CHANNELS, N, p=CH_P))
    mer = rng.choice(MERCHANTS, N)
    shift = pd.Series(ch).map(CH_LEGIT_SHIFT).values
    amt = np.exp(rng.normal(cust['mu'].values[cid] + shift, cust['sig'].values[cid]))
    fraud = np.zeros(N, bool)
    tech = np.full(N, '', dtype=object)

    # ---- fraud initiation (weights from NIBSS channel & month mix) ---------
    ch_share = pd.Series(NIBSS_CH_COUNT) / sum(NIBSS_CH_COUNT.values())
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    m_share = np.array(NIBSS_MONTH_COUNT) / sum(NIBSS_MONTH_COUNT)
    m_factor = m_share / (mdays / 365)
    month = month_of(t)
    # choose initiating events per channel (quota = NIBSS channel share of the target
    # fraud count) by weighted sampling without replacement on month x merchant weights;
    # draw follow-on counts and keep the prefix whose total hits the channel quota
    kept = t >= WARM_S                                  # fraud is injected only into the released year
    target = int(round(prevalence * kept.sum()))
    init = np.zeros(N, bool); n_extra = np.zeros(N, int)
    for c in CHANNELS:
        quota = int(round(ch_share[c] * target))
        pool = np.flatnonzero((ch == c) & kept)
        wc = (m_factor[month[pool] - 1] * pd.Series(mer[pool]).map(MERCHANT_FRAUD_FACTOR).values)
        cand = rng.choice(pool, size=min(len(pool), quota), replace=False, p=wc / wc.sum())
        extra_c = rng.geometric(1 - FRAUD_FOLLOW_P, len(cand)) - 1
        keep = np.cumsum(1 + extra_c) <= quota
        init[cand[keep]] = True; n_extra[cand[keep]] = extra_c[keep]

    # ---- fraud amounts, technique, takeover bursts -------------------------
    loss_mean = {c: NIBSS_CH_LOSS_M[c] * 1e6 / NIBSS_CH_COUNT[c] for c in CHANNELS}
    tp = np.array([NIBSS_TECH_COUNT[k] for k in TECHNIQUES], float); tp /= tp.sum()

    ridx = np.repeat(np.arange(N), n_extra)
    gap = rng.exponential(300.0, len(ridx))
    # cumulative gaps within each initiating event (ridx is sorted)
    cg = np.cumsum(gap)
    if len(ridx):
        fst = np.r_[True, ridx[1:] != ridx[:-1]]
        fi = np.flatnonzero(fst)
        gi = np.cumsum(fst) - 1
        cg = cg - (cg[fi] - gap[fi])[gi]
    t_ex = np.minimum(t[ridx] + cg, TOTAL_S - 1)

    # initiating rows: relabel in place
    fraud[init] = True
    tech[init] = rng.choice(TECHNIQUES, init.sum(), p=tp)
    tm = pd.Series(ch[init]).map(loss_mean).values
    amt[init] = np.exp(rng.normal(np.log(tm) - FRAUD_SIGMA ** 2 / 2, FRAUD_SIGMA))

    # follow-on rows appended
    e_ch, e_tech = ch[ridx], tech[ridx]
    e_tm = pd.Series(e_ch).map(loss_mean).values
    e_amt = np.exp(rng.normal(np.log(e_tm) - FRAUD_SIGMA ** 2 / 2, FRAUD_SIGMA))
    cid = np.r_[cid, cid[ridx]]; t = np.r_[t, t_ex]
    ch = np.r_[ch, e_ch]; mer = np.r_[mer, rng.choice(MERCHANTS, len(ridx))]
    amt = np.r_[amt, e_amt]; fraud = np.r_[fraud, np.ones(len(ridx), bool)]
    tech = np.r_[tech, e_tech]
    N = len(cid)

    # ---- sort chronologically within customer; causal features -------------
    order = np.lexsort((t, cid))
    cid, t, ch, mer, amt, fraud, tech = (a[order] for a in (cid, t, ch, mer, amt, fraud, tech))
    amt = np.round(amt, 2)
    first = np.r_[True, cid[1:] != cid[:-1]]
    gstart = np.flatnonzero(first)
    gid = np.cumsum(first) - 1

    def gcum(x):                                       # within-customer cumulative sum
        c = np.cumsum(x)
        return c - (c[gstart] - x[gstart])[gid]

    # All behavioural features use PRIOR transactions of the same customer only
    # (the current transaction is excluded from every baseline and window).
    cnt = gcum(np.ones(N)) - 1                          # number of prior transactions
    s1, s2 = gcum(amt) - amt, gcum(amt ** 2) - amt ** 2
    has = cnt > 0
    mean_tot = np.where(has, s1 / np.maximum(cnt, 1), amt)        # no history -> current amount
    std_tot = np.where(has, np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, 0)), 0.0)

    key = cid * KEY + t
    P1, P2 = np.r_[0, np.cumsum(amt)], np.r_[0, np.cumsum(amt ** 2)]
    idx = np.arange(N)
    lo24 = np.searchsorted(key, key - 86400, 'left')    # prior rows j with t - t_j <= 24 h, j < i
    lo7 = np.searchsorted(key, key - 7 * 86400, 'left')  # prior rows with t - t_j <= 7 days (exact seconds)
    n24 = idx - lo24; sum24 = P1[idx] - P1[lo24]
    n7 = idx - lo7
    m7 = np.where(n7 > 0, (P1[idx] - P1[lo7]) / np.maximum(n7, 1), mean_tot)
    sd7 = np.where(n7 > 0, np.sqrt(np.maximum((P2[idx] - P2[lo7]) / np.maximum(n7, 1) - ((P1[idx] - P1[lo7]) / np.maximum(n7, 1)) ** 2, 0)), 0.0)

    ratio = np.where(has, amt / np.where(mean_tot > 0, mean_tot, 1), 1.0)
    velocity = np.where(has, n24 * sum24 / np.where(mean_tot > 0, mean_tot, 1), 0.0)
    chdiv = np.zeros(N)
    for c in CHANNELS:
        chdiv += (gcum((ch == c).astype(float)) - (ch == c) > 0)
    online = np.where(has, (gcum(np.isin(ch, ['Mobile', 'Web', 'ECOM']).astype(float))
                            - np.isin(ch, ['Mobile', 'Web', 'ECOM'])) / np.maximum(cnt, 1), 0.0)

    ts = pd.to_datetime(t, unit='s', origin=ORIGIN)
    hr, dw, mo = ts.hour.values, ts.dayofweek.values, ts.month.values
    mf = pd.Series(MERCHANT_FRAUD_FACTOR)
    mrs = ((mf - mf.min()) / (mf.max() - mf.min()) * 0.8 + 0.1).round(6)

    df = pd.DataFrame({
        'customer_id': cust['customer_id'].values[cid],
        'timestamp': ts,
        'amount': amt, 'channel': ch,
        'bank': cust['bank'].values[cid],
        'merchant_category': mer,
        'location': cust['location'].values[cid],
        'age_group': cust['age_group'].values[cid],
        'is_fraud': fraud.astype(int),
        'fraud_technique': np.where(fraud, tech, None),
        'amount_log': np.log1p(amt),
        'amount_vs_mean_ratio': ratio.round(6),
        'amount_sum_24h': sum24.round(2), 'amount_mean_7d': m7.round(2),
        'amount_std_7d': sd7.round(2), 'tx_count_24h': n24,
        'amount_mean_total': mean_tot.round(2), 'amount_std_total': std_tot.round(2),
        'tx_count_total': cnt.astype(int), 'channel_diversity': chdiv.astype(int),
        'online_channel_ratio': online.round(6), 'velocity_score': velocity.round(6),
        'merchant_risk_score': pd.Series(mer).map(mrs).values,
        'hour_sin': np.sin(2 * np.pi * hr / 24), 'hour_cos': np.cos(2 * np.pi * hr / 24),
        'day_sin': np.sin(2 * np.pi * dw / 7), 'day_cos': np.cos(2 * np.pi * dw / 7),
        'month_sin': np.sin(2 * np.pi * mo / 12), 'month_cos': np.cos(2 * np.pi * mo / 12),
        'is_peak_hour': ((hr >= 9) & (hr <= 18)).astype(int),
    })
    if drop_burnin:
        df = df[df['timestamp'] >= '2023-01-01']
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    df.insert(0, 'transaction_id', [f'TXN{i:09d}' for i in range(len(df))])
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--customers', type=int, default=25000)
    ap.add_argument('--prevalence', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    d = generate(a.customers, a.prevalence, a.seed)
    print(f'{len(d):,} rows, {d.is_fraud.sum():,} fraud ({d.is_fraud.mean():.4%})')
    d.to_parquet(f'{a.out}/nibss_synthetic_fraud.parquet', index=False)
