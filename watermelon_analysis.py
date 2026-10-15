import os
import logging
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
for _n in ("cmdstanpy", "prophet"):
    logging.getLogger(_n).setLevel(logging.ERROR)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTDIR = os.path.join(BASE_DIR, "output")
TRADE_FILE = os.path.join(DATA_DIR, "Статистика+внешней+торговли.xlsx")
FX_FILE = os.path.join(DATA_DIR, "Курс.xlsx")

FORECAST_YEARS = [2022, 2023, 2024]
H = len(FORECAST_YEARS)
CV_MIN_TRAIN = 9
Z80, Z95 = 1.2816, 1.96
RANDOM_STATE = 42

os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

KEYS = ["imp_qty_kg", "imp_value_kusd", "exp_qty_kg", "exp_value_kusd"]
RU = {"imp_qty_kg": "Импорт арбузов, кг",
      "imp_value_kusd": "Импорт арбузов, тыс. USD",
      "exp_qty_kg": "Экспорт арбузов, кг",
      "exp_value_kusd": "Экспорт арбузов, тыс. USD"}


def _to_num(s):
    if isinstance(s, (int, float, np.number)):
        return float(s)
    if s is None:
        return np.nan
    t = (str(s).replace("\xa0", "").replace(" ", "")
         .replace("\u2009", "").replace(",", "."))
    try:
        return float(t)
    except ValueError:
        return np.nan


def load_trade(path):
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        hdr = None
        for i in range(min(40, len(raw))):
            row = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if any(v.startswith("направление") for v in row):
                hdr = i
                break
        if hdr is None:
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=hdr)
        df.columns = [str(c).strip() for c in df.columns]

        def col(sub):
            for c in df.columns:
                if sub.lower() in c.lower():
                    return c
            return None

        c_dir, c_year = col("направление"), col("период")
        c_qty = col("количество") or col("кг")
        c_val = col("долларов") or col("тысяч")
        if any(c is None for c in [c_dir, c_year, c_qty, c_val]):
            continue
        out = df[[c_dir, c_year, c_qty, c_val]].copy()
        out.columns = ["direction", "year", "qty_kg", "value_kusd"]
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out = out.dropna(subset=["year", "direction"])
        out["year"] = out["year"].astype(int)
        out["qty_kg"] = out["qty_kg"].map(_to_num)
        out["value_kusd"] = out["value_kusd"].map(_to_num)
        out["direction"] = out["direction"].astype(str).str.strip().str.capitalize()
        out = out[out["direction"].isin(["Экспорт", "Импорт"])]
        print(f"[OK] Торговые данные: лист '{sheet}', строк: {len(out)}")
        return out.reset_index(drop=True)
    raise ValueError("Не найдена таблица внешней торговли.")


def load_fx(path):
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for i in range(min(40, len(raw))):
            if str(raw.iloc[i, 0]).strip().lower() == "год":
                sub = raw.iloc[i + 1:, [0, 1]].copy()
                sub.columns = ["year", "fx"]
                sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
                sub["fx"] = sub["fx"].map(_to_num)
                sub = sub.dropna().astype({"year": int})
                fx = sub.set_index("year")["fx"].sort_index()
                fx.index.name = "year"
                print(f"[OK] Курс BYN/USD: лист '{sheet}', {fx.index.min()}–{fx.index.max()}")
                return fx
    raise ValueError("Не найдена таблица курсов (строка 'Год').")


def build_series(trade):
    y0, y1 = int(trade["year"].min()), int(trade["year"].max())
    full = pd.Index(range(y0, y1 + 1), name="year")
    out = {}
    for direction, tag in [("Импорт", "imp"), ("Экспорт", "exp")]:
        g = (trade[trade["direction"] == direction]
             .groupby("year")[["qty_kg", "value_kusd"]].sum().reindex(full))
        miss = g.index[g["qty_kg"].isna()].tolist()
        if miss:
            print(f"[i] {direction}: нет записей за {miss} -> поставок не было, заполнено 0")
        g = g.fillna(0.0)
        out[f"{tag}_qty_kg"] = g["qty_kg"].astype(float).rename(f"{tag}_qty_kg")
        out[f"{tag}_value_kusd"] = g["value_kusd"].astype(float).rename(f"{tag}_value_kusd")
    return out


def make_transform(use_log):
    if use_log:
        f = lambda x: np.log1p(np.asarray(x, float))
        inv = lambda z: np.maximum(np.expm1(np.asarray(z, float)), 0.0)
    else:
        f = lambda x: np.asarray(x, float)
        inv = lambda z: np.maximum(np.asarray(z, float), 0.0)
    return f, inv


def metrics(y_true, y_pred, y_train):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom > 0, np.abs(e) / np.where(denom > 0, denom, 1), 0)) * 100
    scale = np.mean(np.abs(np.diff(np.asarray(y_train, float))))
    mase = mae / scale if scale > 0 else np.nan
    return dict(MAE=mae, RMSE=rmse, sMAPE=smape, MASE=mase)


def dominant_period(x, pmin=2, pmax=6):
    x = np.asarray(x, float) - np.mean(x)
    f, P = signal.periodogram(x, fs=1.0, detrend=False)
    f, P = f[1:], P[1:]
    per = 1.0 / f
    sel = (per >= pmin) & (per <= pmax)
    if not sel.any():
        return None
    return float(per[sel][np.argmax(P[sel])])


def periodic_component(x, p):
    n = len(x)
    pos = np.arange(n) % p
    means = np.array([x[pos == j].mean() for j in range(p)])
    means -= means.mean()
    return means[pos]


def decompose(series, name, use_log):
    tf, _ = make_transform(use_log)
    y = tf(series.values)
    n = len(y)
    cycle, trend = hpfilter(y, lamb=6.25)
    trend_loess = lowess(y, np.arange(n), frac=0.6, return_sorted=False)
    p_raw = dominant_period(cycle)
    p_int = int(round(p_raw)) if p_raw else None
    if p_int and 2 <= p_int <= n // 2:
        seas = periodic_component(cycle, p_int)
    else:
        seas, p_int = np.zeros(n), None
    resid = cycle - seas
    vy = np.var(y)
    sh = lambda c: float(np.var(c) / vy) if vy > 0 else np.nan
    return dict(name=name, use_log=use_log, years=series.index.values,
                raw=series.values.astype(float), y=y, trend=trend, trend_loess=trend_loess,
                cycle=cycle, seasonal=seas, resid=resid, period=p_int,
                shares=dict(trend=sh(trend), cycle=sh(cycle), seasonal=sh(seas), resid=sh(resid)),
                r2_trend=1 - sh(cycle))


def plot_decomposition(d, fname):
    yrs = d["years"]
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(yrs, d["raw"], "o-", color="k")
    ax[0].set_title(f'{d["name"]}: исходный ряд')
    lbl = "log1p(y)" if d["use_log"] else "y (уровни)"
    ax[1].plot(yrs, d["y"], "o-", label=lbl)
    ax[1].plot(yrs, d["trend"], "r-", lw=2, label="тренд HP (λ=6.25)")
    ax[1].plot(yrs, d["trend_loess"], "g--", lw=1.2, label="тренд LOESS (контроль)")
    ax[1].legend(fontsize=7)
    ax[1].set_title(f'Тренд: доля дисперсии {d["shares"]["trend"]:.0%}, '
                    f'R² тренда {d["r2_trend"]:.0%}')
    ax[2].axhline(0, color="grey", lw=.8)
    ax[2].plot(yrs, d["cycle"], "o-", color="tab:purple")
    ax[2].set_title(f'Циклическая компонента (y − тренд), доля дисперсии {d["shares"]["cycle"]:.0%}')
    ax[3].axhline(0, color="grey", lw=.8)
    ax[3].plot(yrs, d["seasonal"], "o-", color="tab:orange")
    ax[3].set_title(f'Квазисезонная компонента, период {d["period"]} г., доля {d["shares"]["seasonal"]:.0%}'
                    if d["period"] else "Квазисезонная компонента не выявлена")
    ax[4].axhline(0, color="grey", lw=.8)
    ax[4].stem(yrs, d["resid"])
    ax[4].set_title(f'Остатки, доля дисперсии {d["shares"]["resid"]:.0%}')
    fig.savefig(os.path.join(OUTDIR, fname))
    plt.close(fig)


def _pack(mean, lo80=None, hi80=None, lo95=None, hi95=None, info=""):
    a = lambda v: None if v is None else np.asarray(v, float)
    return dict(mean=np.asarray(mean, float), lo80=a(lo80), hi80=a(hi80),
                lo95=a(lo95), hi95=a(hi95), info=info)


def m_naive(train, h, use_log):
    tf, inv = make_transform(use_log)
    y = tf(train.values)
    sd = np.std(np.diff(y), ddof=1)
    se = sd * np.sqrt(np.arange(1, h + 1))
    m = np.repeat(y[-1], h)
    return _pack(inv(m), inv(m - Z80 * se), inv(m + Z80 * se),
                 inv(m - Z95 * se), inv(m + Z95 * se), "Naive (случайное блуждание)")


def m_ets(train, h, use_log):
    tf, inv = make_transform(use_log)
    y = tf(train.values)
    n = len(y)
    best = None
    for trend, damped, lbl in [(None, False, "ETS(A,N,N)"), ("add", False, "ETS(A,A,N)"),
                               ("add", True, "ETS(A,Ad,N)")]:
        try:
            fit = ETSModel(y, error="add", trend=trend, damped_trend=damped,
                           initialization_method="estimated").fit(disp=False)
            if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                best = (fit.aic, lbl, fit)
        except Exception:
            continue
    if best is None:
        return None
    aic, lbl, fit = best
    try:
        pred = fit.get_prediction(start=n, end=n + h - 1)
        s80, s95 = pred.summary_frame(alpha=0.20), pred.summary_frame(alpha=0.05)
        return _pack(inv(s80["mean"].values), inv(s80["pi_lower"].values), inv(s80["pi_upper"].values),
                     inv(s95["pi_lower"].values), inv(s95["pi_upper"].values),
                     f"{lbl} ({'log1p' if use_log else 'уровни'}), AIC={aic:.1f}")
    except Exception:
        m = np.asarray(fit.forecast(h), float)
        se = np.nanstd(np.asarray(fit.resid, float), ddof=1) * np.sqrt(np.arange(1, h + 1))
        return _pack(inv(m), inv(m - Z80 * se), inv(m + Z80 * se),
                     inv(m - Z95 * se), inv(m + Z95 * se), f"{lbl}, AIC={aic:.1f} (прибл. ДИ)")


def _choose_d(y):
    try:
        p = adfuller(y, maxlag=1, autolag=None, regression="c")[1]
        return 0 if p < 0.05 else 1
    except Exception:
        return 1


def m_arima(train, h, use_log):
    tf, inv = make_transform(use_log)
    y = tf(train.values)
    d = _choose_d(y)
    trends = ["c"] if d == 0 else ["n", "c"]
    best = None
    for (p, q), tr in itertools.product([(p, q) for p in range(3) for q in range(3) if p + q <= 2], trends):
        try:
            fit = SARIMAX(y, order=(p, d, q), trend=tr).fit(disp=False, maxiter=200)
            if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                best = (fit.aic, (p, d, q), tr, fit)
        except Exception:
            continue
    if best is None:
        return None
    aic, order, tr, fit = best
    fc = fit.get_forecast(h)
    m = np.asarray(fc.predicted_mean, float)
    c80, c95 = np.asarray(fc.conf_int(alpha=0.20)), np.asarray(fc.conf_int(alpha=0.05))
    return _pack(inv(m), inv(c80[:, 0]), inv(c80[:, 1]), inv(c95[:, 0]), inv(c95[:, 1]),
                 f"ARIMA{order} trend='{tr}' ({'log1p' if use_log else 'уровни'}), AIC={aic:.1f}")


def m_prophet(train, h, use_log):
    if not HAS_PROPHET:
        return None
    tf, inv = make_transform(use_log)
    df = pd.DataFrame({"ds": pd.to_datetime([f"{int(y)}-01-01" for y in train.index]),
                       "y": tf(train.values)})
    fut_years = range(int(train.index[-1]) + 1, int(train.index[-1]) + 1 + h)
    fut = pd.DataFrame({"ds": pd.to_datetime([f"{y}-01-01" for y in fut_years])})
    try:
        mdl = Prophet(growth="linear", n_changepoints=min(3, max(0, len(df) - 4)),
                      changepoint_prior_scale=0.1, yearly_seasonality=False,
                      weekly_seasonality=False, daily_seasonality=False,
                      interval_width=0.80, uncertainty_samples=1000)
        mdl.fit(df)
        mean = mdl.predict(fut)["yhat"].values
        samp = mdl.predictive_samples(fut)["yhat"]
        q = lambda a: np.percentile(samp, a, axis=1)
        return _pack(inv(mean), inv(q(10)), inv(q(90)), inv(q(2.5)), inv(q(97.5)),
                     f"Prophet (linear, {'log1p' if use_log else 'уровни'})")
    except Exception:
        return None


def _lag_matrix(d, n_lags):
    X = np.array([d[i - n_lags:i] for i in range(n_lags, len(d))], float)
    return X, d[n_lags:]


def _ml_recursive(train, h, use_log, make_reg, label, n_lags=2):
    tf, inv = make_transform(use_log)
    y = tf(train.values)
    d = np.diff(y)
    if len(d) < n_lags + 4:
        return None
    X, Y = _lag_matrix(d, n_lags)
    reg = make_reg()
    reg.fit(X, Y)
    hist, lv, out = list(d), y[-1], []
    for _ in range(h):
        dk = float(reg.predict(np.array(hist[-n_lags:], float).reshape(1, -1))[0])
        hist.append(dk)
        lv += dk
        out.append(lv)
    return _pack(inv(np.array(out)), info=f"{label} на лагах Δ, рекурсивно; ДИ по ошибкам CV")


def reg_boost():
    if HAS_LGB:
        return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=3,
                                 min_child_samples=2, min_data_in_bin=1,
                                 random_state=RANDOM_STATE, verbose=-1)
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                     random_state=RANDOM_STATE)


BOOST_NAME = "LightGBM" if HAS_LGB else "GradientBoosting"


def model_zoo():
    zoo = {"Naive": m_naive, "ETS": m_ets, "ARIMA": m_arima}
    if HAS_PROPHET:
        zoo["Prophet"] = m_prophet
    zoo[f"{BOOST_NAME}(ML)"] = lambda tr, h, ul: _ml_recursive(tr, h, ul, reg_boost, BOOST_NAME)
    return zoo


def rolling_cv(series, zoo, use_log, min_train=CV_MIN_TRAIN, horizon=H):
    n = len(series)
    rows, preds = [], []
    for name, f in zoo.items():
        for k in range(min_train, n):
            tr, h = series.iloc[:k], min(horizon, n - k)
            te = series.iloc[k:k + h]
            try:
                r = f(tr, h, use_log)
            except Exception:
                r = None
            if r is None:
                continue
            m = metrics(te.values, r["mean"], tr.values)
            m.update(model=name, origin=int(tr.index[-1]), h=h)
            rows.append(m)
            for i, yr in enumerate(te.index):
                preds.append(dict(model=name, origin=int(tr.index[-1]), year=int(yr), step=i + 1,
                                  y_true=float(te.values[i]), y_pred=float(r["mean"][i])))
    cv = pd.DataFrame(rows)
    if cv.empty:
        raise RuntimeError("CV не дала результатов")
    agg = cv.groupby("model").agg(folds=("MASE", "size"), MAE=("MAE", "mean"),
                                  RMSE=("RMSE", "mean"), sMAPE=("sMAPE", "mean"),
                                  MASE=("MASE", "mean"))
    agg["MASE/Naive"] = agg["MASE"] / agg.loc["Naive", "MASE"]
    return agg.sort_values(["MASE", "RMSE"]), cv, pd.DataFrame(preds)


def choose_best(agg):
    full = agg[agg["folds"] == agg["folds"].max()]
    return full.sort_values(["MASE", "RMSE"]).index[0]


def add_cv_intervals(r, cv_pred, model, use_log, h):
    tf, inv = make_transform(use_log)
    sub = cv_pred[cv_pred["model"] == model]
    e = tf(sub["y_true"].values) - tf(sub["y_pred"].values)
    step = sub["step"].values
    sd_all = np.sqrt(np.mean(e ** 2))
    sds = []
    for k in range(1, h + 1):
        ek = e[step == k]
        sd = np.sqrt(np.mean(ek ** 2)) if len(ek) >= 2 else sd_all * np.sqrt(k)
        if sds:
            sd = max(sd, sds[-1])
        sds.append(sd)
    se = np.array(sds)
    m = tf(r["mean"])
    r.update(lo80=inv(m - Z80 * se), hi80=inv(m + Z80 * se),
             lo95=inv(m - Z95 * se), hi95=inv(m + Z95 * se))
    return r


def adf_row(x, name):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    try:
        res = adfuller(x, maxlag=1, autolag=None, regression="c")
        return dict(ряд=name, ADF=res[0], p=res[1], n=res[3],
                    вывод="стационарен (p<0.10)" if res[1] < 0.10 else "нестационарен")
    except Exception:
        return dict(ряд=name, ADF=np.nan, p=np.nan, n=len(x), вывод="не оценено")


def corr_block(series, fx, use_log_map):
    yrs = sorted(set(series[KEYS[0]].index) & set(fx.index))
    f = fx.reindex(yrs).values.astype(float)
    lf, dlf = np.log(f), np.diff(np.log(f))
    t = np.arange(len(yrs))

    adfs = [adf_row(lf, "log FX"), adf_row(dlf, "Δlog FX")]
    corr_rows, reg_rows, data = [], [], {}
    for k in KEYS:
        tf, _ = make_transform(use_log_map[k])
        y = series[k].reindex(yrs).values.astype(float)
        ty, dty = tf(y), np.diff(tf(y))
        lbl = "log1p" if use_log_map[k] else "уровни"
        data[k] = dict(y=y, ty=ty, dty=dty)
        adfs += [adf_row(ty, f"{k} ({lbl})"), adf_row(dty, f"Δ {k}")]

        for kind, a, b in [("уровни", y, f),
                           (f"первые разности: Δ{lbl}(Y) vs Δlog(FX)", dty, dlf)]:
            pr, pp = stats.pearsonr(a, b)
            sr, sp = stats.spearmanr(a, b)
            corr_rows.append(dict(показатель=k, преобразование=kind, n=len(a),
                                  Pearson_r=pr, p_Pearson=pp, Spearman_rho=sr, p_Spearman=sp))

        m1 = sm.OLS(ty, sm.add_constant(np.column_stack([lf, t]))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1}, use_t=True)
        reg_rows.append(dict(показатель=k, спецификация=f"{lbl}(Y) ~ log(FX) + тренд",
                             beta_FX=m1.params[1], t=m1.tvalues[1], p=m1.pvalues[1], R2=m1.rsquared))
    return (pd.DataFrame(corr_rows), pd.DataFrame(reg_rows),
            pd.DataFrame(adfs), data, dict(years=yrs, f=f, lf=lf, dlf=dlf))


def plot_raw(series, fx):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, k in zip(ax.ravel(), KEYS):
        a.plot(series[k].index, series[k].values, "o-")
        a.set_title(RU[k]); a.set_xlabel("год")
    fig.suptitle("Исходные данные: ТН ВЭД 080711 «Арбузы свежие», Республика Беларусь, 2008–2021")
    fig.savefig(os.path.join(OUTDIR, "01_raw_series.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fx.index, fx.values, "s-", color="tab:red")
    ax.axvspan(2021.5, 2024.5, color="grey", alpha=0.12, label="период прогноза")
    ax.set_title("Средний официальный курс BYN/USD, 2008–2024"); ax.set_xlabel("год"); ax.legend()
    fig.savefig(os.path.join(OUTDIR, "02_fx.png")); plt.close(fig)


def plot_forecast(s, key, best, fcs, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s.index, s.values, "o-", color="k", lw=2, label="факт 2008–2021")
    colors = plt.cm.tab10.colors
    for i, (nm, r) in enumerate(fcs.items()):
        is_best = nm == best
        xs = [s.index[-1]] + FORECAST_YEARS
        ys = [s.values[-1]] + list(r["mean"])
        ax.plot(xs, ys, "-" if is_best else "--", marker="o", lw=2.4 if is_best else 1.0,
                alpha=1.0 if is_best else 0.6, color=colors[i % 10],
                label=f"{nm}{' ★ лучшая' if is_best else ''}")
    b = fcs[best]
    if b["lo95"] is not None:
        ax.fill_between(FORECAST_YEARS, b["lo95"], b["hi95"], alpha=0.15, color="tab:blue",
                        label="95% ДИ (лучшая)")
    if b["lo80"] is not None:
        ax.fill_between(FORECAST_YEARS, b["lo80"], b["hi80"], alpha=0.30, color="tab:blue",
                        label="80% ДИ (лучшая)")
    ax.axvline(2021.5, color="grey", ls=":")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{RU[key]}: факт и прогноз на 2022–2024 гг.")
    ax.set_xlabel("год"); ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(OUTDIR, fname)); plt.close(fig)


def _heat(ax, c, title):
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title); ax.grid(False)
    plt.colorbar(im, ax=ax)


def plot_corr(data, fxd):
    lev = pd.DataFrame({k: data[k]["y"] for k in KEYS}); lev["FX"] = fxd["f"]
    dif = pd.DataFrame({k: data[k]["dty"] for k in KEYS}); dif["Δlog FX"] = fxd["dlf"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    _heat(ax[0], lev.corr(method="spearman"), "Уровни: Спирмен (общий тренд -> ложная связь)")
    _heat(ax[1], dif.corr(), "Первые разности: Пирсон (связь после снятия тренда)")
    fig.savefig(os.path.join(OUTDIR, "07_corr_heatmaps.png")); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    y = data["imp_value_kusd"]["y"]
    ax[0].scatter(fxd["f"], y)
    for x_, y_, yr in zip(fxd["f"], y, fxd["years"]):
        ax[0].annotate(str(yr), (x_, y_), fontsize=6)
    ax[0].set_xlabel("курс BYN/USD"); ax[0].set_ylabel("импорт, тыс. USD")
    ax[0].set_title("Уровни: импорт vs курс (оба ряда трендовые)")
    ax[1].scatter(fxd["dlf"], data["imp_value_kusd"]["dty"], color="tab:green")
    ax[1].axhline(0, color="grey", lw=.8); ax[1].axvline(0, color="grey", lw=.8)
    ax[1].set_xlabel("Δlog(курс)"); ax[1].set_ylabel("Δlog1p(импорт, тыс. USD)")
    ax[1].set_title("Приращения: связь после устранения тренда")
    fig.savefig(os.path.join(OUTDIR, "08_scatter_fx.png")); plt.close(fig)


def main():
    print("=" * 79)
    print("ВНЕШНЯЯ ТОРГОВЛЯ РБ, ТН ВЭД 080711 «АРБУЗЫ СВЕЖИЕ»: ДЕКОМПОЗИЦИЯ, ПРОГНОЗ, КУРС")
    print("=" * 79)
    trade = load_trade(TRADE_FILE)
    fx = load_fx(FX_FILE)
    series = build_series(trade)
    use_log = {k: bool(series[k].min() > 0) for k in KEYS}
    print("Преобразование рядов:", {RU[k]: ("log1p" if v else "уровни") for k, v in use_log.items()})

    desc = pd.DataFrame(series).describe().T[["mean", "std", "min", "max"]]
    desc["CV_%"] = desc["std"] / desc["mean"] * 100
    desc["CAGR_%"] = [((series[k].values[-1] / series[k].values[0]) ** (1 / (len(series[k]) - 1)) - 1) * 100
                      if series[k].values[0] > 0 else np.nan for k in KEYS]
    print("\n--- Описательная статистика, 2008–2021 ---\n", desc.round(2))
    plot_raw(series, fx)

    print("\n" + "=" * 79 + "\nДЕКОМПОЗИЦИЯ: y = тренд(HP) + цикл;  цикл = квазисезонная + остаток\n" + "=" * 79)
    dec, dec_rows = {}, []
    for i, k in enumerate(KEYS, 1):
        d = decompose(series[k], RU[k], use_log[k])
        dec[k] = d
        plot_decomposition(d, f"03_{i}_decomposition_{k}.png")
        dec_rows.append(dict(показатель=RU[k], преобразование="log1p" if use_log[k] else "уровни",
                             R2_тренда=d["r2_trend"], период_квазисезонности_лет=d["period"],
                             доля_тренд=d["shares"]["trend"], доля_цикл=d["shares"]["cycle"],
                             доля_квазисезон=d["shares"]["seasonal"], доля_остаток=d["shares"]["resid"]))
    dec_tbl = pd.DataFrame(dec_rows)
    print(dec_tbl.round(3).to_string(index=False))

    print("\n" + "=" * 79 + "\nПРОГНОЗ НА 2022–2024 (rolling-origin CV, обучение >= %d лет, h <= %d)\n"
          % (CV_MIN_TRAIN, H) + "=" * 79)
    zoo = model_zoo()
    print("Модели:", ", ".join(zoo))
    all_cv, all_fc, summary = {}, {}, []
    for k in KEYS:
        s = series[k]
        agg, cv_raw, cv_pred = rolling_cv(s, zoo, use_log[k])
        best = choose_best(agg)
        print(f"\n### {RU[k]} ({'log1p' if use_log[k] else 'уровни'})")
        print(agg.round(3).to_string())
        print(f"ЛУЧШАЯ по MASE: {best}  (MASE/Naive = {agg.loc[best, 'MASE/Naive']:.2f})")

        fcs = {}
        for nm, f in zoo.items():
            try:
                r = f(s, H, use_log[k])
            except Exception:
                r = None
            if r is None:
                continue
            if r["lo80"] is None:
                r = add_cv_intervals(r, cv_pred, nm, use_log[k], H)
            fcs[nm] = r
        all_cv[k], all_fc[k] = agg, fcs
        plot_forecast(s, k, best, fcs, f"04_forecast_{k}.png")
        b = fcs[best]
        for i, yr in enumerate(FORECAST_YEARS):
            summary.append(dict(показатель=RU[k], модель=best, год=yr, прогноз=b["mean"][i],
                                ДИ80_низ=b["lo80"][i], ДИ80_верх=b["hi80"][i],
                                ДИ95_низ=b["lo95"][i], ДИ95_верх=b["hi95"][i],
                                MASE_CV=agg.loc[best, "MASE"], описание=b["info"]))
    summary = pd.DataFrame(summary)
    print("\n--- ИТОГОВЫЙ ПРОГНОЗ (лучшие модели) ---")
    print(summary.drop(columns="описание").round(1).to_string(index=False))

    print("\n" + "=" * 79 + "\nКОРРЕЛЯЦИЯ СТОИМОСТИ ИМПОРТА/ЭКСПОРТА С КУРСОМ BYN/USD\n" + "=" * 79)
    corr_tbl, reg_tbl, adf_tbl, cdata, fxd = corr_block(series, fx, use_log)
    print(f"Курс собран за {fx.index.min()}–{fx.index.max()}; корреляция считается на пересечении "
          f"с торговыми данными: {fxd['years'][0]}–{fxd['years'][-1]} (n={len(fxd['years'])}).")
    print("\n--- ADF (maxlag=1; на 14 точках мощность теста низкая) ---\n",
          adf_tbl.round(3).to_string(index=False))
    print("\n--- Коэффициенты корреляции ---\n", corr_tbl.round(3).to_string(index=False))
    print("\n--- Регрессии (HAC/Newey–West, t-распределение) ---\n", reg_tbl.round(3).to_string(index=False))
    plot_corr(cdata, fxd)

    xlsx = os.path.join(OUTDIR, "results.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame(series).to_excel(w, sheet_name="Ряды")
        fx.rename("BYN_USD").to_frame().to_excel(w, sheet_name="Курс")
        desc.to_excel(w, sheet_name="Дескриптивы")
        dec_tbl.to_excel(w, sheet_name="Декомпозиция", index=False)
        for k in KEYS:
            d = dec[k]
            pd.DataFrame({"y": d["y"], "trend_hp": d["trend"], "cycle": d["cycle"],
                          "quasi_seasonal": d["seasonal"], "resid": d["resid"]},
                         index=d["years"]).to_excel(w, sheet_name=f"DEC_{k[:25]}")
        for k in KEYS:
            all_cv[k].to_excel(w, sheet_name=f"CV_{k[:26]}")
            pd.DataFrame({nm: r["mean"] for nm, r in all_fc[k].items()},
                         index=FORECAST_YEARS).to_excel(w, sheet_name=f"FC_{k[:26]}")
        summary.to_excel(w, sheet_name="Прогноз_итог", index=False)
        corr_tbl.to_excel(w, sheet_name="Корреляции", index=False)
        reg_tbl.to_excel(w, sheet_name="Регрессии", index=False)
        adf_tbl.to_excel(w, sheet_name="ADF", index=False)
    print(f"\n[OK] Таблицы: {xlsx}\n[OK] Графики: {OUTDIR}{os.sep}*.png\nГОТОВО.")


if __name__ == "__main__":
    main()