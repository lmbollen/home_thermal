"""`home-thermal` command line: check a config, get data, fit, cross-validate, simulate.

Durations (--warmup, --block) are pandas strings such as 24h, 48h, 3D.
Set HOME_THERMAL_DEBUG=1 to see full tracebacks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from home_thermal.core import (
    DATA_TERM_PARAMS,
    Network,
    check_names,
    coupling,
    default_free,
    energy_balance,
    load_house,
    parameter_set,
    read_csv,
    required_columns,
    sensor_temps,
    simulate,
    simulate_detailed,
)


def _free(cfg, args):
    extra = [n for n in args.free.split(",") if n]
    fix = [n for n in args.fix.split(",") if n]
    check_names(cfg, dict.fromkeys(extra + fix))
    free = default_free(cfg) + [n for n in extra if n not in default_free(cfg)]
    return [n for n in free if n not in fix]


def _params(cfg, path):
    if not path:
        return {}
    from home_thermal.calibrate import load_params
    return load_params(path, cfg)


def _out_path(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_check(args):
    cfg = load_house(args.house)
    ps = parameter_set(cfg)
    free = default_free(cfg)
    em = cfg["emitters"]
    print(f"{cfg['name']}: {len(cfg['zones'])} zones, {len(em['radiators'])} radiators, "
          f"{len(em['floor_heating'])} floor-heating zones")
    net = Network(cfg)
    print(f"nodes: {net.n}; H_house at priors: {net.H_house():.0f} W/K (party wall {net.H_parts['party']:.0f} W/K)")
    print("sensors:", ", ".join(zc["sensor"] for zc in cfg["zones"].values() if zc["sensor"]))
    req = ["T_water (or T_flow + T_return)" if c == "T_water" else c for c in required_columns(cfg)]
    print("required data columns:", ", ".join(req))
    optional = [cfg["solar"]["proxy"]] if "g_sol" in ps else []
    if cfg["boiler"] is not None:
        b = cfg["boiler"]
        optional += [f"{b['gas_column']} (gas, {b['gas_unit']})"] + ([b["dhw_flag"]] if b["dhw_flag"] else [])
    else:
        optional += ["HP_elec_W (+T_flow)"]
    optional += ["HP_heat_W", "HP_heat_per_K"]
    print("optional data columns:", ", ".join(optional))
    print(f"parameters: {len(ps)} ({len(free)} fitted by default; * = only with the matching "
          "heat-source data: heat meter, electricity or gas)")
    width = max(map(len, ps))
    for n, p in ps.items():
        mark = ("fit*" if n in DATA_TERM_PARAMS else "fit ") if n in free else "fix "
        print(f"  {mark} {n:{width}s} prior {p.prior:<8g} {p.doc}")


def cmd_demo(args):
    from home_thermal.core.synthetic import demo_house, demo_house_yaml, synthetic_data
    if args.days < 1:
        raise ValueError("--days must be at least 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "house.yaml").write_text(demo_house_yaml(), encoding="utf-8")
    truth = {"m_env": 1.3, "m_vent": 0.8, "Q_int": 250.0, "m_rad": 1.2}
    cfg = demo_house()
    df = synthetic_data(cfg, truth, days=args.days, seed=args.seed)
    df.to_csv(out / "data.csv")
    h_true = Network(cfg, truth).H_house()
    (out / "truth.json").write_text(json.dumps({"params": truth, "H_house_W_per_K": round(h_true, 1),
                                                "heat_pump_eta": 0.5}, indent=2), encoding="utf-8")
    print(f"wrote {out / 'house.yaml'}, {out / 'data.csv'} ({len(df)} rows) and {out / 'truth.json'}")
    print(f"the data was generated with these parameters (others at their prior); true H_house {h_true:.0f} W/K")


def cmd_synth(args):
    from home_thermal.core.synthetic import synthetic_data
    cfg = load_house(args.house)
    truth = json.loads(args.truth) if args.truth else {}
    df = synthetic_data(cfg, truth, days=args.days, seed=args.seed, solar=not args.no_solar)
    df.to_csv(_out_path(args.out))
    print(f"wrote {args.out} ({len(df)} rows); true H_house {Network(cfg, truth).H_house():.0f} W/K")


def cmd_hass_fetch(args):
    from home_thermal.hass import HassConfig, fetch_db
    print(fetch_db(HassConfig.load(args.hass), args.dest))


def cmd_hass_load(args):
    from home_thermal.hass import HassConfig, load_full
    if not Path(args.db).is_file():
        raise FileNotFoundError(f"database not found: {args.db}")
    df = load_full(args.db, HassConfig.load(args.hass), step=args.step)
    df.to_csv(_out_path(args.out))
    print(f"wrote {args.out}: {len(df)} rows, {df.index[0]} .. {df.index[-1]}")


def cmd_simulate(args):
    cfg = load_house(args.house)
    df = read_csv(args.data)
    net = Network(cfg, _params(cfg, args.params))
    X, Q = simulate(net, df)
    out = X.join(sensor_temps(net, X, df).add_prefix("pred_")).assign(heat_W=Q)
    out.to_csv(_out_path(args.out))
    print(f"wrote {args.out}")


def _print_params(cfg, res):
    ps = parameter_set(cfg)
    rows = [{"parameter": n, "prior": ps[n].prior, "fitted": res.params[n],
             "at_bound": "yes" if n in res.at_bound else ""} for n in res.free]
    print(pd.DataFrame(rows).set_index("parameter").round(3).to_string())
    print(f"H_house {Network(cfg, res.params).H_house():.0f} W/K (prior {Network(cfg).H_house():.0f})")
    if res.at_bound:
        print(f"note: {', '.join(res.at_bound)} ended on a bound: the data cannot place them, or they are "
              "compensating for something the model lacks. Consider --fix (see README, 'How far to trust it').")


def cmd_fit(args):
    from home_thermal.calibrate import FitSettings, fit, save_params, score
    cfg = load_house(args.house)
    df = read_csv(args.data)
    settings = FitSettings(warmup=args.warmup)
    res = fit(cfg, df, free=_free(cfg, args), settings=settings, verbose=args.verbose)
    _print_params(cfg, res)
    print(score(cfg, res.params, df, warmup=settings.warmup)[["bias", "rmse"]].round(2).to_string())
    save_params(_out_path(args.out), res.params, sd_laplace_understated=res.sd,
                meta={"free": res.free, "at_bound": res.at_bound, "house": str(args.house), "data": str(args.data)})
    print(f"wrote {args.out}")


def cmd_crossval(args):
    from home_thermal.calibrate import FitSettings, cross_validate, fit, save_params
    cfg = load_house(args.house)
    df = read_csv(args.data)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free = _free(cfg, args)
    settings = FitSettings(warmup=args.warmup)
    cv = cross_validate(cfg, df, free=free, block=args.block, settings=settings, forecasts=not args.no_forecast)
    report = cv.report(cfg)
    print(report)
    cv.folds.to_csv(out / "crossval.csv", index=False)
    cv.params.to_csv(out / "crossval_params.csv", index=False)
    if cv.forecasts is not None:
        cv.forecasts.to_csv(out / "forecast.csv", index=False)
    print("\nFINAL fit on all data:")
    res = fit(cfg, df, free=free, settings=settings)
    _print_params(cfg, res)
    r = cv.room_rmse()
    save_params(out / "params.json", res.params, sd_laplace_understated=res.sd,
                cv_spread=cv.spread(cfg)[["fold_min", "fold_max"]].to_dict("index"),
                meta={"free": res.free, "at_bound": res.at_bound, "house": str(args.house), "data": str(args.data),
                      "data_start": str(df.index[0]), "data_end": str(df.index[-1]),
                      "cv_rmse_fitted_K": float(r["fitted"].mean()), "cv_rmse_prior_K": float(r["prior"].mean())})
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"wrote {out}: report.txt, crossval.csv, crossval_params.csv, "
          f"{'forecast.csv, ' if cv.forecasts is not None else ''}params.json")


def cmd_flows(args):
    cfg = load_house(args.house)
    df = read_csv(args.data)
    net = Network(cfg, _params(cfg, args.params))
    eb = energy_balance(net, simulate_detailed(net, df), start=df.index[0] + pd.Timedelta(args.warmup))
    print(f"over {eb['hours']:.0f} h (after a {args.warmup} warm-up):")
    print("SOURCES (kWh; 'info' rows are not extra heat)\n" + eb["sources"].round(1).to_string(index=False))
    print("\nLOSSES (kWh; negative = heat gained from that boundary)\n"
          + eb["losses"].groupby(["to", "path"]).kWh.sum().round(1).to_string())
    print(f"\nstored: {eb['stored_kWh']:.1f} kWh")
    print("\nWARMING: steady-state temperature rise (K) of the row zone per kW delivered to the column zone\n"
          + coupling(net)["warming"].round(2).to_string())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="home-thermal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(fn=fn)
        return s

    s = p("check", cmd_check, "validate a house config and list its parameters and required data")
    s.add_argument("house")
    s = p("demo", cmd_demo, "write the demo house and synthetic data to a folder")
    s.add_argument("out")
    s.add_argument("--days", type=float, default=12)
    s.add_argument("--seed", type=int, default=0)
    s = p("synth", cmd_synth, "synthetic data for your own house config (known 'true' parameters)")
    s.add_argument("house")
    s.add_argument("out")
    s.add_argument("--truth", default="", help='true parameters as JSON, e.g. \'{"m_env": 1.2}\' (default: priors)')
    s.add_argument("--days", type=float, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-solar", action="store_true", help="no solar proxy column")
    s = p("hass-fetch", cmd_hass_fetch, "copy the Home Assistant recorder DB over ssh (read-only)")
    s.add_argument("hass")
    s.add_argument("--dest", default="data/ha_db")
    s = p("hass-load", cmd_hass_load, "convert a Home Assistant recorder DB into a data CSV")
    s.add_argument("hass")
    s.add_argument("--db", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--step", default="15min", help="time step, e.g. 15min (default)")
    for name, fn, help_ in (("fit", cmd_fit, "fit the parameters to data"),
                            ("crossval", cmd_crossval, "blocked cross-validation, then a final fit")):
        s = p(name, fn, help_)
        s.add_argument("house")
        s.add_argument("data")
        s.add_argument("--fix", default="", help="comma-separated parameters to hold at their prior")
        s.add_argument("--free", default="", help="comma-separated extra parameters to fit")
        s.add_argument("--warmup", default="48h", help="start of data not scored, e.g. 48h (default)")
    s = sub.choices["fit"]
    s.add_argument("--out", default="params.json")
    s.add_argument("--verbose", type=int, default=0)
    s = sub.choices["crossval"]
    s.add_argument("--out-dir", default="results")
    s.add_argument("--block", default="72h", help="length of each held-out block, e.g. 72h (default)")
    s.add_argument("--no-forecast", action="store_true", help="skip the forecast scoring (faster)")
    for name, fn, help_ in (("simulate", cmd_simulate, "simulate with given (or prior) parameters"),
                            ("flows", cmd_flows, "energy balance and room coupling")):
        s = p(name, fn, help_)
        s.add_argument("house")
        s.add_argument("data")
        s.add_argument("--params", default=None, help="params.json from fit/crossval (default: priors)")
    sub.choices["simulate"].add_argument("--out", default="simulated.csv")
    sub.choices["flows"].add_argument("--warmup", default="24h", help="start of data left out, e.g. 24h (default)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # report any failure as one line; full traceback with HOME_THERMAL_DEBUG=1
        if os.environ.get("HOME_THERMAL_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        msg = " ".join(str(msg).split())        # one line, even for multi-line parser messages
        print(f"error: {type(e).__name__}: {msg}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
