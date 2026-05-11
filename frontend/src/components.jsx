// Policy Response Analyser — shared components
// Each component pushed to window so other JSX scripts can use them.

const { useState, useEffect, useMemo, useRef, useCallback } = React;

/* ════════════════════════════════════════════════════════════
   ICONS — small inline SVGs (stroke=1.6, viewBox 24)
   ════════════════════════════════════════════════════════════ */
const Icon = ({ name, size = 14, ...rest }) => {
  const paths = {
    docs: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8",
    search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z M21 21l-4.35-4.35",
    list: "M3 7h18 M3 12h18 M3 17h12",
    chart: "M3 3v18h18 M7 14l4-4 4 4 5-5",
    upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
    download: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
    copy: "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2 M14 2H10a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V8z",
    refresh: "M3 12a9 9 0 0 1 15-6.7L21 8 M21 3v5h-5 M21 12a9 9 0 0 1-15 6.7L3 16 M3 21v-5h5",
    reset: "M3 12a9 9 0 0 1 15-6.7L21 8 M21 3v5h-5 M21 12a9 9 0 0 1-15 6.7L3 16 M3 21v-5h5",
    moon: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
    sun: "M12 1v2 M12 21v2 M4.22 4.22l1.42 1.42 M18.36 18.36l1.42 1.42 M1 12h2 M21 12h2 M4.22 19.78l1.42-1.42 M18.36 5.64l1.42-1.42 M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z",
    chevDown: "M6 9l6 6 6-6",
    chevRight: "M9 6l6 6-6 6",
    chevLeft: "M15 6l-9 6 9 6",
    x: "M18 6L6 18 M6 6l12 12",
    check: "M5 13l4 4L19 7",
    plus: "M12 5v14 M5 12h14",
    filter: "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
    file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6",
    play: "M5 3l14 9-14 9V3z",
    stop: "M5 5h14v14H5z",
    info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 16v-4 M12 8h.01",
    warning: "M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
    warn: "M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
    sparkle: "M12 2v4 M12 18v4 M2 12h4 M18 12h4 M4.93 4.93l2.83 2.83 M16.24 16.24l2.83 2.83 M4.93 19.07l2.83-2.83 M16.24 7.76l2.83-2.83",
    target: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
    book: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20 M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z",
    layers: "M12 2 2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
    flask: "M9 2v6.31L4.93 16.5A2 2 0 0 0 6.7 19.5h10.6a2 2 0 0 0 1.77-3L15 8.31V2 M8 2h8 M8 14h8",
    grid: "M3 3h7v7H3z M14 3h7v7h-7z M14 14h7v7h-7z M3 14h7v7H3z",
    panel: "M3 3h18v18H3z M9 3v18 M3 12h6",
    arrowRight: "M5 12h14 M12 5l7 7-7 7",
    spark: "M12 2l2.5 6L21 9.3l-5 4.4L17.5 21 12 17.5 6.5 21 8 13.7l-5-4.4L9.5 8z",
    lock: "M5 11h14v10H5z M8 11V7a4 4 0 0 1 8 0v4",
    open: "M3 9l4-6h10l4 6 M3 9v12h18V9 M3 9h18 M9 13h6",
    terminal: "M4 17l3-3-3-3 M11 17h9",
    link: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71 M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
    rec: "M3 7h18 M3 12h18 M3 17h12",
  };
  const d = paths[name] || "";
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" {...rest}>
      {d.split(" M").map((part, i) => (
        <path key={i} d={(i === 0 ? part : "M" + part)} />
      ))}
    </svg>
  );
};
window.Icon = Icon;

/* ════════════════════════════════════════════════════════════
   SCREEN HEADER
   ════════════════════════════════════════════════════════════ */
const ScreenHeader = ({ title, subtitle, meta }) => (
  <div className="screen-header">
    <div className="h-text">
      <h1>{title}</h1>
      {subtitle && <div className="h-sub">{subtitle}</div>}
    </div>
    {meta && <div className="h-meta">{meta}</div>}
  </div>
);
window.ScreenHeader = ScreenHeader;

/* ════════════════════════════════════════════════════════════
   STAGE CHIP — for the recommendation workspace status
   ════════════════════════════════════════════════════════════ */
const StageChip = ({ variant = "lock", children }) => (
  <span className={"stage-chip " + variant}>
    {variant === "run"  && <span className="spinner" style={{width:9, height:9, borderWidth:1.4}}/>}
    {variant === "lock" && <Icon name="lock" size={10}/>}
    {variant === "done" && <Icon name="check" size={11}/>}
    {children}
  </span>
);
window.StageChip = StageChip;

/* ════════════════════════════════════════════════════════════
   STATUS BADGE (matched / weak / none / no_response)
   ════════════════════════════════════════════════════════════ */
const StatusBadge = ({ status }) => {
  const map = {
    matched:     { label: "Matched",     icon: "check"   },
    weak:        { label: "Weak match",  icon: "warning" },
    none:        { label: "No response", icon: "info"    },
    no_response: { label: "No response", icon: "info"    },
  };
  const m = map[status] || map.none;
  const cls = status === "none" ? "no_response" : (status || "none");
  return (
    <span className={"status-badge " + cls}>
      <Icon name={m.icon} size={10}/>
      {m.label}
    </span>
  );
};
window.StatusBadge = StatusBadge;

/* ════════════════════════════════════════════════════════════
   CLASSIFICATION PILL — 5 variants
   ════════════════════════════════════════════════════════════ */
const _CL_LABEL = {
  accepted: "Accepted", partial: "Partial", rejected: "Rejected",
  not_addressed: "Not addressed", pending: "Pending",
};
const _CL_FULL = {
  accepted: "Accepted", partial: "Partially accepted", rejected: "Rejected",
  not_addressed: "Not addressed", pending: "Pending",
};
const ClassificationPill = ({ label, compact }) => (
  <span className={"classification-pill " + (label || "pending") + (compact ? " compact" : "")}>
    <span className="dot" style={{width:6, height:6, background:"currentColor", borderRadius:999, opacity:0.6}}/>
    {compact ? _CL_LABEL[label] : _CL_FULL[label]}
  </span>
);
window.ClassificationPill = ClassificationPill;

/* ════════════════════════════════════════════════════════════
   METHOD TAG
   ════════════════════════════════════════════════════════════ */
const _METHOD_LABEL = {
  exact_label: "exact",
  structure: "structure",
  semantic: "semantic",
  chunk_fallback: "fallback",
  sequence_correction: "sequence",
};
const MethodTag = ({ m }) => (
  <span className={"method-tag " + m}>{_METHOD_LABEL[m] || m}</span>
);
window.MethodTag = MethodTag;

/* ════════════════════════════════════════════════════════════
   CONFIDENCE BADGE — bar + numeric, green→amber→red gradient
   ════════════════════════════════════════════════════════════ */
const ConfidenceBadge = ({ value }) => {
  const v = value ?? 0;
  const pct = Math.max(0, Math.min(1, v)) * 100;
  const color = v >= 0.8 ? "var(--ok)"
              : v >= 0.6 ? "var(--accent)"
              : v >= 0.4 ? "var(--warn)" : "var(--err)";
  return (
    <span className="conf-badge">
      <span className="bar"><span className="fill" style={{width: pct + "%", background: color}}/></span>
      <span className="num">{v.toFixed(2)}</span>
    </span>
  );
};
window.ConfidenceBadge = ConfidenceBadge;

/* ════════════════════════════════════════════════════════════
   EMPTY STATE
   ════════════════════════════════════════════════════════════ */
const EmptyState = ({ icon = "docs", title = "Nothing here", message, action }) => (
  <div className="empty-state">
    <div className="icon-circle"><Icon name={icon} size={16}/></div>
    <div style={{color:"var(--ink-2)", fontWeight:600, fontSize:13.5, marginBottom:4}}>{title}</div>
    {message && <div style={{maxWidth:360, margin:"0 auto"}}>{message}</div>}
    {action && <div style={{marginTop:14}}>{action}</div>}
  </div>
);
window.EmptyState = EmptyState;

/* ════════════════════════════════════════════════════════════
   SECTION LABEL — small caps section divider
   ════════════════════════════════════════════════════════════ */
const SectionLabel = ({ children, count }) => (
  <div className="section-label">
    <span>{children}</span>
    <span className="grow-line"></span>
    {count !== undefined && <span className="count">{count}</span>}
  </div>
);
window.SectionLabel = SectionLabel;

/* ════════════════════════════════════════════════════════════
   HTML TEXT — renders pre-marked HTML safely
   ════════════════════════════════════════════════════════════ */
const HtmlText = ({ html, className }) => (
  <span className={className} dangerouslySetInnerHTML={{__html: html}}/>
);
window.HtmlText = HtmlText;
