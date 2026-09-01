/**
 * Dispersion Desk — operator console.
 *
 * Four views, matching the four questions someone watching the desk will ask:
 *   Dashboard      what is the book worth, and what risk is it carrying?
 *   Agent Activity what is the agent doing right now?
 *   Decisions      why did it do that, and how did it turn out?
 *   Risk Centre    what did it refuse to do, and why?
 *
 * The UI computes nothing. Every number shown is produced and journalled by the
 * backend first, so the screen and the audit trail cannot disagree.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const API = "http://127.0.0.1:8000";

type Tab = "dashboard" | "activity" | "decisions" | "risk";

interface Status {
  agent_running: boolean;
  propose_only: boolean;
  kill_switch: boolean;
  paper_trading: boolean;
  has_alpaca_credentials: boolean;
  llm_provider: string;
  options_feed: string;
  index_symbol: string;
  basket: string[];
  basket_coverage_pct: number;
  weights_as_of: string;
  weights_age_days: number;
  correlation_premium_entry: number;
  limits: Record<string, number>;
}

interface Dashboard {
  net_asset_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  market_is_open: boolean;
  greeks: { delta: number; gamma: number; vega: number; theta: number };
  limits: Record<string, number>;
  open_defined_risk: number;
  open_defined_risk_pct: number;
  risk_by_underlying: Record<string, number>;
  positions: Array<Record<string, string | null>>;
}

interface Activity {
  id: number;
  at: string;
  step: string;
  level: string;
  message: string;
}

interface Decision {
  basket_id: string;
  decided_at: string;
  direction: string;
  approved: number;
  max_loss: number;
  rationale: string;
  memo: string;
}

const money = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const signed = (n: number, digits = 0) =>
  `${n >= 0 ? "+" : ""}${n.toLocaleString("en-US", { maximumFractionDigits: digits })}`;
const clock = (iso: string) => (iso ? new Date(iso).toLocaleTimeString("en-GB") : "--");

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`);
  return r.json();
}

/** A limit shown as a bar. Amber past 80% of the cap, red once breached. */
function LimitBar({
  label,
  value,
  limit,
  unit = "",
}: {
  label: string;
  value: number;
  limit: number;
  unit?: string;
}) {
  const used = limit > 0 ? Math.abs(value) / limit : 0;
  const tone = used >= 1 ? "bad" : used >= 0.8 ? "warn" : "ok";
  return (
    <div className="limit">
      <div className="limit-head">
        <span>{label}</span>
        <span className={`mono ${tone}`}>
          {signed(value, 1)}
          {unit} / {limit}
          {unit}
        </span>
      </div>
      <div className="bar">
        <div className={`fill ${tone}`} style={{ width: `${Math.min(used, 1) * 100}%` }} />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  sub,
}: {
  label: string;
  value: string;
  tone?: string;
  sub?: string;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ""}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [status, setStatus] = useState<Status | null>(null);
  const [dash, setDash] = useState<Dashboard | null>(null);
  const [dashError, setDashError] = useState("");
  const [activity, setActivity] = useState<Activity[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [risk, setRisk] = useState<{ rejected: Decision[]; limits: Record<string, number> } | null>(
    null,
  );
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const feedRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getJSON<Status>("/api/status"));
    } catch {
      /* best-effort: the banner already signals a stale console */
    }
    try {
      setDash(await getJSON<Dashboard>("/api/dashboard"));
      setDashError("");
    } catch (e) {
      setDashError(e instanceof Error ? e.message : String(e));
    }
    try {
      setDecisions(await getJSON<Decision[]>("/api/decisions"));
      setRisk(await getJSON("/api/risk"));
    } catch {
      /* an empty journal on first run is not an error */
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 15000);
    return () => clearInterval(t);
  }, [refresh]);

  // Live agent feed over SSE rather than polling, so each step appears the
  // moment the agent journals it.
  useEffect(() => {
    void getJSON<Activity[]>("/api/activity?limit=100")
      .then((rows) => setActivity(rows.reverse()))
      .catch(() => undefined);
    const es = new EventSource(`${API}/api/activity/stream`);
    es.onmessage = (e) => setActivity((prev) => [...prev.slice(-400), JSON.parse(e.data)]);
    return () => es.close();
  }, []);

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
  }, [activity]);

  const post = async (path: string, label: string) => {
    setBusy(true);
    try {
      const r = await fetch(`${API}${path}`, { method: "POST" });
      const body = await r.json().catch(() => ({}));
      setToast(
        r.ok
          ? `${label}: ${body.outcome ?? body.detail ?? "done"}`
          : `${label} failed: ${body.detail}`,
      );
      await refresh();
    } catch (e) {
      setToast(`${label} failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setBusy(false);
      setTimeout(() => setToast(""), 6000);
    }
  };

  const openDecision = async (id: string) => {
    try {
      setSelected(await getJSON(`/api/decisions/${id}`));
    } catch {
      setToast("Could not load that decision.");
    }
  };

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>Dispersion Desk</h1>
          <span className="tagline">Trading the gap between index volatility and its parts</span>
        </div>
        <div className="badges">
          {status?.paper_trading && <span className="badge ok">PAPER</span>}
          {status?.propose_only && <span className="badge warn">PROPOSE&nbsp;ONLY</span>}
          {status?.kill_switch && <span className="badge bad">KILL&nbsp;SWITCH</span>}
          <span className={`badge ${status?.agent_running ? "live" : "idle"}`}>
            {status?.agent_running ? "AGENT LIVE" : "AGENT IDLE"}
          </span>
        </div>
        <div className="controls">
          <button disabled={busy} onClick={() => void post("/api/cycle/run", "Cycle")}>
            Run cycle now
          </button>
          {status?.agent_running ? (
            <button disabled={busy} onClick={() => void post("/api/agent/stop", "Agent")}>
              Stop agent
            </button>
          ) : (
            <button disabled={busy} onClick={() => void post("/api/agent/start", "Agent")}>
              Start agent
            </button>
          )}
          <button
            className="danger"
            disabled={busy}
            onClick={() =>
              void post(`/api/kill-switch?engaged=${!status?.kill_switch}`, "Kill switch")
            }
          >
            {status?.kill_switch ? "Release kill switch" : "Kill switch"}
          </button>
        </div>
      </header>

      {status && !status.has_alpaca_credentials && (
        <div className="alert">
          Alpaca credentials are not configured. Copy <code>.env.example</code> to <code>.env</code>{" "}
          and add the keys from a fresh paper trading account.
        </div>
      )}
      {toast && <div className="toast">{toast}</div>}

      <nav>
        {(["dashboard", "activity", "decisions", "risk"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {
              {
                dashboard: "Dashboard",
                activity: "Agent Activity",
                decisions: "Decisions",
                risk: "Risk Centre",
              }[t]
            }
          </button>
        ))}
      </nav>

      <main>
        {tab === "dashboard" && (
          <>
            {dashError && <div className="alert">{dashError}</div>}
            <section className="stats">
              <Stat label="Portfolio value" value={dash ? money(dash.net_asset_value) : "--"} />
              <Stat
                label="Daily P&L"
                value={dash ? money(dash.daily_pnl) : "--"}
                tone={dash && dash.daily_pnl >= 0 ? "ok" : "bad"}
                sub={dash ? `${signed(dash.daily_pnl_pct, 2)}%` : undefined}
              />
              <Stat
                label="Net delta"
                value={dash ? signed(dash.greeks.delta) : "--"}
                tone={
                  dash && Math.abs(dash.greeks.delta) <= (dash.limits.max_net_delta ?? 0)
                    ? "ok"
                    : "bad"
                }
                sub="direction-neutral target"
              />
              <Stat
                label="Vega"
                value={dash ? signed(dash.greeks.vega) : "--"}
                sub="the risk the desk chooses"
              />
              <Stat label="Theta / day" value={dash ? signed(dash.greeks.theta) : "--"} />
              <Stat
                label="Capital at risk"
                value={dash ? money(dash.open_defined_risk) : "--"}
                sub={dash ? `${dash.open_defined_risk_pct.toFixed(2)}% of NAV` : undefined}
              />
            </section>

            <section className="panel">
              <h2>Risk envelope</h2>
              {dash ? (
                <div className="limits">
                  <LimitBar
                    label="Net delta"
                    value={dash.greeks.delta}
                    limit={dash.limits.max_net_delta}
                  />
                  <LimitBar
                    label="Portfolio vega"
                    value={dash.greeks.vega}
                    limit={dash.limits.max_portfolio_vega}
                  />
                  <LimitBar
                    label="Portfolio gamma"
                    value={dash.greeks.gamma}
                    limit={dash.limits.max_portfolio_gamma}
                  />
                  <LimitBar
                    label="Daily theta"
                    value={dash.greeks.theta}
                    limit={dash.limits.max_daily_theta}
                  />
                  <LimitBar
                    label="Total defined risk"
                    value={dash.open_defined_risk_pct}
                    limit={dash.limits.max_total_risk_pct}
                    unit="%"
                  />
                </div>
              ) : (
                <p className="muted">Waiting for the broker.</p>
              )}
            </section>

            <section className="panel">
              <h2>Positions</h2>
              {dash?.positions.length ? (
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Qty</th>
                      <th>Class</th>
                      <th>Market value</th>
                      <th>Unrealised</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dash.positions.map((p) => (
                      <tr key={String(p.symbol)}>
                        <td className="mono">{p.symbol}</td>
                        <td className="mono">{p.qty}</td>
                        <td>{p.asset_class}</td>
                        <td className="mono">{p.market_value}</td>
                        <td className={`mono ${Number(p.unrealized_pl) >= 0 ? "ok" : "bad"}`}>
                          {p.unrealized_pl}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="muted">No open positions.</p>
              )}
            </section>

            {status && (
              <section className="panel">
                <h2>Strategy configuration</h2>
                <div className="kv">
                  <div>
                    <span>Index</span>
                    <b>{status.index_symbol}</b>
                  </div>
                  <div>
                    <span>Basket</span>
                    <b>{status.basket.join(", ")}</b>
                  </div>
                  <div>
                    <span>Index weight covered</span>
                    <b>{status.basket_coverage_pct}%</b>
                  </div>
                  <div>
                    <span>Entry threshold</span>
                    <b>±{status.correlation_premium_entry} correlation</b>
                  </div>
                  <div>
                    <span>Options feed</span>
                    <b>{status.options_feed}</b>
                  </div>
                  <div>
                    <span>LLM provider</span>
                    <b>{status.llm_provider}</b>
                  </div>
                  <div>
                    <span>Weights as of</span>
                    <b>
                      {status.weights_as_of} ({status.weights_age_days}d old)
                    </b>
                  </div>
                </div>
                <p className="muted small">
                  The basket covers {status.basket_coverage_pct}% of the index by weight. The
                  remainder is a known basis error, documented rather than hidden.
                </p>
              </section>
            )}
          </>
        )}

        {tab === "activity" && (
          <section className="panel">
            <h2>Agent activity</h2>
            <div className="feed" ref={feedRef}>
              {activity.length === 0 && (
                <p className="muted">Nothing yet. Run a cycle to watch the agent work.</p>
              )}
              {activity.map((a) => (
                <div key={a.id} className={`feed-row ${a.level}`}>
                  <span className="mono time">{clock(a.at)}</span>
                  <span className="step">{a.step}</span>
                  <span className="msg">{a.message}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {tab === "decisions" && (
          <section className="panel">
            <h2>Decisions</h2>
            {decisions.length === 0 && <p className="muted">No decisions journalled yet.</p>}
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Basket</th>
                  <th>Direction</th>
                  <th>Verdict</th>
                  <th>Max loss</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {decisions.map((d) => (
                  <tr key={d.basket_id}>
                    <td className="mono">{clock(d.decided_at)}</td>
                    <td className="mono">{d.basket_id}</td>
                    <td>{d.direction.replace(/_/g, " ")}</td>
                    <td className={d.approved ? "ok" : "bad"}>
                      {d.approved ? "APPROVED" : "REJECTED"}
                    </td>
                    <td className="mono">{money(d.max_loss)}</td>
                    <td>
                      <button onClick={() => void openDecision(d.basket_id)}>Detail</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {tab === "risk" && risk && (
          <section className="panel">
            <h2>Rejected trades</h2>
            <p className="muted">
              Refusals are the evidence the controls are real. Every rejection below carries the
              arithmetic that produced it.
            </p>
            {risk.rejected.length === 0 && <p className="muted">Nothing rejected yet.</p>}
            {risk.rejected.map((d) => (
              <div key={d.basket_id} className="reject">
                <div className="reject-head">
                  <b className="mono">{d.basket_id}</b>
                  <span className="mono">{clock(d.decided_at)}</span>
                </div>
                <div className="muted">{d.rationale}</div>
                <div className="memo">{d.memo}</div>
                <button onClick={() => void openDecision(d.basket_id)}>Full risk report</button>
              </div>
            ))}
          </section>
        )}
      </main>

      {selected && (
        <div className="modal" onClick={() => setSelected(null)}>
          <div className="modal-body" onClick={(e) => e.stopPropagation()}>
            <button className="close" onClick={() => setSelected(null)}>
              ×
            </button>
            <h2 className="mono">{String(selected.basket_id)}</h2>
            <p>{String(selected.memo ?? "")}</p>

            <h3>Risk checks</h3>
            <table>
              <thead>
                <tr>
                  <th>Gate</th>
                  <th>Result</th>
                  <th>Observed</th>
                  <th>Limit</th>
                </tr>
              </thead>
              <tbody>
                {(selected.checks as Array<Record<string, unknown>> | undefined)?.map((c, i) => (
                  <tr key={i}>
                    <td className="mono">{String(c.name)}</td>
                    <td className={c.passed ? "ok" : "bad"}>{c.passed ? "PASS" : "FAIL"}</td>
                    <td className="mono">
                      {c.observed === null ? "--" : Number(c.observed).toFixed(3)}
                    </td>
                    <td className="mono">
                      {c.limit_value === null ? "--" : Number(c.limit_value).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {(selected.attributions as Array<Record<string, number>> | undefined)?.length ? (
              <>
                <h3>P&amp;L attribution</h3>
                <p className="muted small">
                  This desk claims its profits come from volatility, not direction. The split below
                  is how that claim gets checked.
                </p>
                {(selected.attributions as Array<Record<string, number | string>>).map((a, i) => (
                  <div key={i} className="attr">
                    {(
                      [
                        "delta_pnl",
                        "gamma_pnl",
                        "vega_pnl",
                        "theta_pnl",
                        "slippage",
                        "residual",
                      ] as const
                    ).map((k) => (
                      <div key={k}>
                        <span>{k.replace("_pnl", "")}</span>
                        <b className={Number(a[k]) >= 0 ? "ok" : "bad"}>{money(Number(a[k]))}</b>
                      </div>
                    ))}
                    <div className="attr-total">
                      <span>total</span>
                      <b>{money(Number(a.total))}</b>
                    </div>
                    <div className="attr-driver">
                      driven by <b>{String(a.dominant)}</b>
                    </div>
                  </div>
                ))}
              </>
            ) : null}

            <h3>Legs</h3>
            <table>
              <thead>
                <tr>
                  <th>Contract</th>
                  <th>Side</th>
                  <th>Qty</th>
                  <th>Price</th>
                  <th>IV</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {(selected.legs as Array<Record<string, unknown>> | undefined)?.map((l, i) => (
                  <tr key={i}>
                    <td className="mono">{String(l.symbol)}</td>
                    <td className={l.side === "buy" ? "ok" : "warn"}>{String(l.side)}</td>
                    <td className="mono">{String(l.quantity)}</td>
                    <td className="mono">{Number(l.price).toFixed(2)}</td>
                    <td className="mono">{(Number(l.implied_volatility) * 100).toFixed(1)}%</td>
                    <td className="mono">{Number(l.delta).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
