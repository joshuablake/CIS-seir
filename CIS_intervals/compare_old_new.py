"""Compare the refit (this directory's results) with the v1.0.0 run.

Inputs for the v1.0.0 run come from re-running compute_diagnostics.py and
export_summary_stats.py on it with the DIAG_*/SEIR_* environment overrides
(see README.md), written to OLD_DIR; OLD_PARAMS is its exported params.csv.gz.
"""
import pandas as pd, numpy as np, os
from config import RESULTS_DIR, FOR_PAPER_DIR
HERE=os.path.dirname(os.path.abspath(__file__))
OLD_DIR=os.environ.get("OLD_DIR", os.path.join(HERE,"reports","old_reanalysis"))
OLD_SUM=os.path.join(OLD_DIR,"incidence_summary_statistics.csv")
NEW_SUM=os.path.join(FOR_PAPER_DIR,"incidence_summary_statistics.csv")
OLD_PAR=os.environ["OLD_PARAMS"]
NEW_PAR=os.path.join(FOR_PAPER_DIR,"params.csv.gz")
OLD_DIAG=os.path.join(OLD_DIR,"all_diagnostics.csv")
NEW_DIAG=os.path.join(RESULTS_DIR,"all_diagnostics.csv")
OLD_CC=os.path.join(OLD_DIR,"chain_checks.csv")
NEW_CC=os.path.join(RESULTS_DIR,"chain_checks.csv")
pd.set_option("display.width",220); pd.set_option("display.max_columns",30); pd.set_option("display.max_rows",200)
expit=lambda x:1/(1+np.exp(-x))
def par(path):
    d=pd.read_csv(path); d=d[d.iteration>=1_000_000]
    out={}
    for name,f in [("psir",lambda v:v),("i0",expit),("matrix_modifiers",np.exp),("theta",np.exp),("beta[0]",np.exp)]:
        g=d[d.parameter==name].groupby("region").value
        out[name]=g.apply(lambda v:"%.3g (%.3g–%.3g)"%tuple(np.quantile(f(v.values),[.5,.025,.975])))
    return pd.DataFrame(out)
print("### PARAMS (final cutoff, post burn-in) — sample->natural: i0 expit, cs exp, theta(rho) exp, rw_sd exp")
po,pn=par(OLD_PAR),par(NEW_PAR)
for c in po.columns: print("\n--",c); print(pd.DataFrame({"old":po[c],"new":pn[c]}))
o=pd.read_csv(OLD_SUM,parse_dates=["day","cutoff_date"]); n=pd.read_csv(NEW_SUM,parse_dates=["day","cutoff_date"])
print("\ncolumns",list(o.columns)); fin=o.cutoff_date.max()
print("\n### ENGLAND gold-standard incidence (median, %) old vs new at selected dates")
for d in ["2020-08-31","2020-09-07","2020-09-14","2020-09-21","2020-09-28","2020-10-05","2020-10-12","2020-10-18","2020-11-01","2020-12-21","2021-01-17"]:
    a=o[(o.region=="England")&(o.cutoff_date==fin)&(o.day==d)]; b=n[(n.region=="England")&(n.cutoff_date==fin)&(n.day==d)]
    if len(a) and len(b): print(d, "old %.4f (%.4f–%.4f)  new %.4f (%.4f–%.4f)  ratio %.2f"%(100*a['median'].iat[0],100*a.q025.iat[0],100*a.q975.iat[0],100*b['median'].iat[0],100*b.q025.iat[0],100*b.q975.iat[0],b['median'].iat[0]/a['median'].iat[0]))
for lab,df in [("old",o),("new",n)]:
    g=df[(df.region=="England")&(df.cutoff_date==fin)&(df.day>="2020-08-31")&(df.day<="2020-09-30")]
    print(lab,"Sept cumulative incidence (sum of daily medians, %%) %.3f"%(100*g['median'].sum()))
def nowcast(df):
    rows=[]
    for reg in sorted(df.region.unique()):
        gs=df[(df.region==reg)&(df.cutoff_date==fin)].set_index("day")
        for c in sorted(df.cutoff_date.unique()):
            if c==fin: continue
            nc=df[(df.region==reg)&(df.cutoff_date==c)].set_index("day")
            j=nc.join(gs,rsuffix="_g",how="inner")
            ov=(j.q975>=j.q025_g)&(j.q025<=j.q975_g)
            lag=None
            for L in range(0,60):
                ok=ov[j.index<=c-pd.Timedelta(days=L)]
                if ok.all(): lag=L; break
            # relative median error at c-7 and c-14
            def rel(L):
                d=c-pd.Timedelta(days=L)
                return j.loc[d,"median"]/j.loc[d,"median_g"] if d in j.index else np.nan
            rows.append(dict(region=reg,cutoff=c.date(),min_lag_all_overlap=lag,ratio_lag0=rel(0),ratio_lag7=rel(7),ratio_lag14=rel(14)))
    return pd.DataFrame(rows)
no,nn=nowcast(o),nowcast(n)
m=no.merge(nn,on=["region","cutoff"],suffixes=("_old","_new"))
print("\n### NOWCAST England: smallest lag L (days) s.t. nowcast & gold-standard 95% CrIs overlap on every day up to cutoff-L; median ratio nowcast/gold at lag 0/7/14")
print(m[m.region=="England"].round(2).to_string(index=False))
print("\n### NOWCAST regions: count of (region,cutoff) with min_lag <=14 / <=7 ; mean |log ratio| at lag 7")
for s in ["old","new"]:
    r=m[m.region!="England"]
    print(s, (r[f"min_lag_all_overlap_{s}"]<=14).sum(), (r[f"min_lag_all_overlap_{s}"]<=7).sum(), len(r), "%.3f"%np.nanmean(np.abs(np.log(r[f"ratio_lag7_{s}"]))))
print("\nregional min lag by cutoff (old/new):")
r=m[m.region!="England"]
print(r.pivot(index="region",columns="cutoff",values="min_lag_all_overlap_old").to_string())
print(r.pivot(index="region",columns="cutoff",values="min_lag_all_overlap_new").to_string())
print("\n### DIAGNOSTICS")
for lab,p,cc in [("old",OLD_DIAG,OLD_CC),("new",NEW_DIAG,NEW_CC)]:
    d=pd.read_csv(p); c=pd.read_csv(cc)
    rh=[x for x in d.columns if "rhat" in x.lower()][0]; es=[x for x in d.columns if "ess" in x.lower() and "bulk" in x.lower()] or [x for x in d.columns if "ess" in x.lower()]
    es=es[0]
    print(lab,"cols",list(d.columns)[:10])
    print(lab,"n",len(d),"max rhat %.3f"%d[rh].max(),"frac rhat>1.05 %.4f"%(d[rh]>1.05).mean(),"frac rhat>1.01 %.3f"%(d[rh]>1.01).mean(),"min ess %.0f"%d[es].min(),"frac ess<400 %.4f"%(d[es]<400).mean())
    print(d.sort_values(rh,ascending=False).head(6)[[x for x in d.columns if x in ("region","cutoff_date","parameter",rh,es)]].to_string(index=False))
    print(lab,"chains",len(c),"invalid",int(c.flag_invalid.sum()),"low_rho",int(c.flag_low_rho.sum()),"min post log rho %.2f"%c.min_log_rho_post.min())
