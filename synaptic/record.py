"""The shared record: every plan, tool call, result, review and finding lives here.

This is the paper's "version-controlled record readable by every agent" plus the curator's
knowledge base. SQLite in WAL mode so the dashboard can read while the harness writes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns(
  id TEXT PRIMARY KEY, name TEXT, brief TEXT, mode TEXT, status TEXT,
  started REAL, ended REAL, workdir TEXT);
CREATE TABLE IF NOT EXISTS stages(
  campaign_id TEXT, idx INTEGER, key TEXT, name TEXT, status TEXT, PRIMARY KEY(campaign_id, idx));
CREATE TABLE IF NOT EXISTS thresholds(
  campaign_id TEXT, key TEXT, value REAL, rationale TEXT, locked_at REAL, PRIMARY KEY(campaign_id, key));
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, campaign_id TEXT, parent_id TEXT, stage TEXT, kind TEXT, title TEXT,
  brief TEXT, meta TEXT, status TEXT, origin TEXT, depth INTEGER, revisions INTEGER DEFAULT 0,
  created REAL, started REAL, ended REAL, summary TEXT, verdict TEXT, triage_note TEXT);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, campaign_id TEXT, task_id TEXT, role TEXT, model TEXT, started REAL, ended REAL,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, calls INTEGER DEFAULT 0, data_chars INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT, session_id TEXT, task_id TEXT, role TEXT,
  ts REAL, type TEXT, content TEXT, meta TEXT);
CREATE INDEX IF NOT EXISTS ev_task ON events(task_id);
CREATE INDEX IF NOT EXISTS ev_camp ON events(campaign_id, id);
CREATE TABLE IF NOT EXISTS kb(
  id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT, task_id TEXT, kind TEXT, title TEXT,
  body TEXT, tags TEXT, created REAL);
CREATE TABLE IF NOT EXISTS candidates(
  id TEXT, campaign_id TEXT, task_id TEXT, family TEXT, variable TEXT, cls TEXT,
  lat REAL, lon REAL, n_cells INTEGER, onset INTEGER, effect REAL, pvalue REAL, qvalue REAL,
  stats TEXT, filters TEXT, passed INTEGER DEFAULT 0, promoted INTEGER DEFAULT 0, outcome TEXT, note TEXT,
  PRIMARY KEY(campaign_id, id));
CREATE TABLE IF NOT EXISTS reports(
  id TEXT PRIMARY KEY, campaign_id TEXT, task_id TEXT, family TEXT, title TEXT, body TEXT,
  figures TEXT, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS games(
  id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT, a TEXT, b TEXT, scores TEXT, winner TEXT);
CREATE TABLE IF NOT EXISTS ratings(
  campaign_id TEXT, report_id TEXT, rating REAL, wins INTEGER, played INTEGER, rank INTEGER,
  PRIMARY KEY(campaign_id, report_id));
CREATE TABLE IF NOT EXISTS hypotheses(
  id TEXT, campaign_id TEXT, question TEXT, title TEXT, statement TEXT, test TEXT, kill TEXT, data TEXT,
  priority REAL, origin TEXT, status TEXT, result TEXT, evidence TEXT, novelty TEXT, review TEXT,
  task_ids TEXT, created REAL, updated REAL, PRIMARY KEY(campaign_id, id));
CREATE TABLE IF NOT EXISTS bench_attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, bench_id TEXT, model TEXT, level INTEGER, case_id TEXT,
  detected INTEGER, onset_ok INTEGER, type_ok INTEGER, location_ok INTEGER, artifact_ok INTEGER,
  score REAL, data_chars INTEGER, longest_run INTEGER, submission TEXT, created REAL);
"""


from .verdicts import CODE_ACTORS, CODE_KILL_TAGS  # noqa: E402


def _strip_code_tags(new: str, old: str) -> str:
    """Neutralise code-check tags an agent adds to a note ("[independent check]" -> "(independent check)"). Tags
    already in the stored note were written by code and are kept when the new note extends it."""
    def clean(s: str) -> str:
        for t in CODE_KILL_TAGS:
            s = s.replace(t, "(" + t[1:-1] + ")")
        return s
    if old and new.startswith(old):
        return old + clean(new[len(old):])
    out = clean(new)
    for t in CODE_KILL_TAGS:  # a rewritten note keeps the code tags the stored note already had
        if t in old and t not in out:
            out = t + " " + out
    return out


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


class Record:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        with self.conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)
            cols = {r[1] for r in c.execute("PRAGMA table_info(hypotheses)").fetchall()}
            for col in ("confirm", "confirmation", "novelty_basis"):  # v4: pre-registered held-out confirmation
                if col not in cols:
                    c.execute(f"ALTER TABLE hypotheses ADD COLUMN {col} TEXT DEFAULT ''")

    @staticmethod
    def _migrate(c):
        """Family ids (e.g. T2M_MAX:shift+:01) repeat across campaigns: old databases keyed candidates on the
        id alone, so a new campaign overwrote an older campaign's families. Re-key on (campaign_id, id)."""
        sql = (c.execute("SELECT sql FROM sqlite_master WHERE name='candidates'").fetchone() or [""])[0] or ""
        if "PRIMARY KEY(campaign_id, id)" in sql.replace("  ", " "):
            return
        c.executescript("""
        ALTER TABLE candidates RENAME TO candidates_old;
        CREATE TABLE candidates(
          id TEXT, campaign_id TEXT, task_id TEXT, family TEXT, variable TEXT, cls TEXT,
          lat REAL, lon REAL, n_cells INTEGER, onset INTEGER, effect REAL, pvalue REAL, qvalue REAL,
          stats TEXT, filters TEXT, passed INTEGER DEFAULT 0, promoted INTEGER DEFAULT 0, outcome TEXT, note TEXT,
          PRIMARY KEY(campaign_id, id));
        INSERT OR REPLACE INTO candidates SELECT * FROM candidates_old;
        DROP TABLE candidates_old;""")
        c.commit()

    # -- connection handling -------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.c = c
        return c

    def x(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            c = self.conn()
            cur = c.execute(sql, tuple(args))
            c.commit()
            return cur

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, tuple(args)).fetchall()]

    def one(self, sql: str, args: Iterable[Any] = ()) -> dict | None:
        r = self.conn().execute(sql, tuple(args)).fetchone()
        return dict(r) if r else None

    # -- campaign --------------------------------------------------------------
    def create_campaign(self, name: str, brief: dict, mode: str, workdir: str) -> str:
        cid = new_id("c_")
        self.x("INSERT INTO campaigns VALUES(?,?,?,?,?,?,?,?)",
               (cid, name, json.dumps(brief), mode, "running", time.time(), None, workdir))
        return cid

    def set_campaign_status(self, cid: str, status: str):
        self.x("UPDATE campaigns SET status=?, ended=? WHERE id=?",
               (status, time.time() if status in ("done", "failed", "stopped") else None, cid))

    def set_stages(self, cid: str, stages: list[tuple[str, str]]):
        for i, (key, name) in enumerate(stages):
            self.x("INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?)", (cid, i, key, name, "pending"))

    def set_stage_status(self, cid: str, key: str, status: str):
        self.x("UPDATE stages SET status=? WHERE campaign_id=? AND key=?", (status, cid, key))

    # -- pre-registered thresholds (immutable once locked) ---------------------
    def lock_thresholds(self, cid: str, thresholds: dict[str, tuple[float, str]]):
        existing = {r["key"] for r in self.q("SELECT key FROM thresholds WHERE campaign_id=?", (cid,))}
        now = time.time()
        for k, (v, why) in thresholds.items():
            if k in existing:
                continue  # never overwrite a locked threshold
            self.x("INSERT INTO thresholds VALUES(?,?,?,?,?)", (cid, k, float(v), why, now))

    def thresholds(self, cid: str) -> dict[str, float]:
        return {r["key"]: r["value"] for r in self.q("SELECT key,value FROM thresholds WHERE campaign_id=?", (cid,))}

    # -- tasks -----------------------------------------------------------------
    def add_task(self, cid: str, *, title: str, brief: str, stage: str, kind: str, origin: str,
                 parent_id: str | None = None, depth: int = 0, meta: dict | None = None,
                 status: str = "queued") -> str:
        tid = new_id("t_")
        self.x("""INSERT INTO tasks(id,campaign_id,parent_id,stage,kind,title,brief,meta,status,origin,depth,created)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
               (tid, cid, parent_id, stage, kind, title, brief, json.dumps(meta or {}), status, origin, depth,
                time.time()))
        return tid

    def update_task(self, tid: str, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        self.x(f"UPDATE tasks SET {cols} WHERE id=?", (*vals, tid))

    def task(self, tid: str) -> dict | None:
        t = self.one("SELECT * FROM tasks WHERE id=?", (tid,))
        if t:
            t["meta"] = json.loads(t["meta"] or "{}")
        return t

    def tasks(self, cid: str, **where) -> list[dict]:
        sql = "SELECT * FROM tasks WHERE campaign_id=?"
        args: list[Any] = [cid]
        for k, v in where.items():
            sql += f" AND {k}=?"
            args.append(v)
        rows = self.q(sql + " ORDER BY created", args)
        for r in rows:
            r["meta"] = json.loads(r["meta"] or "{}")
        return rows

    # -- sessions & events -----------------------------------------------------
    def open_session(self, cid: str, tid: str | None, role: str, model: str) -> str:
        sid = new_id("s_")
        self.x("INSERT INTO sessions(id,campaign_id,task_id,role,model,started) VALUES(?,?,?,?,?,?)",
               (sid, cid, tid, role, model, time.time()))
        return sid

    def close_session(self, sid: str):
        self.x("UPDATE sessions SET ended=? WHERE id=?", (time.time(), sid))

    def add_usage(self, sid: str, tin: int, tout: int, data_chars: int = 0):
        self.x("UPDATE sessions SET tokens_in=tokens_in+?, tokens_out=tokens_out+?, calls=calls+1, "
               "data_chars=data_chars+? WHERE id=?", (tin, tout, data_chars, sid))

    def add_data_chars(self, sid: str, n: int):
        self.x("UPDATE sessions SET data_chars=data_chars+? WHERE id=?", (n, sid))

    def event(self, cid: str, sid: str | None, tid: str | None, role: str, type_: str, content: str,
              meta: dict | None = None):
        self.x("INSERT INTO events(campaign_id,session_id,task_id,role,ts,type,content,meta) VALUES(?,?,?,?,?,?,?,?)",
               (cid, sid, tid, role, time.time(), type_, content, json.dumps(meta or {})))

    # -- knowledge base ----------------------------------------------------------
    def kb_add(self, cid: str, tid: str | None, kind: str, title: str, body: str, tags: list[str] | None = None):
        self.x("INSERT INTO kb(campaign_id,task_id,kind,title,body,tags,created) VALUES(?,?,?,?,?,?,?)",
               (cid, tid, kind, title, body, ",".join(tags or []), time.time()))

    # notes that later turned out wrong because a service became available: shown, but marked superseded
    STALE_MARKERS = ("unavailable", "not available", "http 401", "cannot be downloaded", "could not download",
                     "no download tool", "requires earthdata credentials", "download was blocked", "not downloadable")

    def kb_search(self, cid: str, query: str, limit: int = 8) -> list[dict]:
        rows = self.q("SELECT * FROM kb WHERE campaign_id=? ORDER BY id DESC", (cid,))
        for r in rows:
            low = f"{r['title']} {r['body']}".lower()
            if r["kind"] in ("method", "caveat", "rejection") and any(m in low for m in self.STALE_MARKERS):
                r["title"] = "[SUPERSEDED where it claims data are unavailable: the Service status line is current] " + r["title"]
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return rows[:limit]
        scored = []
        for r in rows:
            text = f"{r['title']} {r['body']} {r['tags']}".lower()
            s = sum(text.count(t) for t in terms)
            if s:
                scored.append((s, r))
        scored.sort(key=lambda z: -z[0])
        return [r for _, r in scored[:limit]]

    # -- candidates ------------------------------------------------------------
    def upsert_candidate(self, cid: str, row: dict):
        cols = ["id", "campaign_id", "task_id", "family", "variable", "cls", "lat", "lon", "n_cells", "onset",
                "effect", "pvalue", "qvalue", "stats", "filters", "passed", "promoted", "outcome", "note"]
        row = {**row, "campaign_id": cid}
        for k in ("stats", "filters"):
            if isinstance(row.get(k), (dict, list)):
                row[k] = json.dumps(row[k])
        self.x(f"INSERT OR REPLACE INTO candidates({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
               [row.get(c) for c in cols])

    def candidates(self, cid: str, **where) -> list[dict]:
        sql = "SELECT * FROM candidates WHERE campaign_id=?"
        args: list[Any] = [cid]
        for k, v in where.items():
            sql += f" AND {k}=?"
            args.append(v)
        rows = self.q(sql, args)
        for r in rows:
            r["stats"] = json.loads(r["stats"] or "{}")
            r["filters"] = json.loads(r["filters"] or "{}")
        return rows

    def update_candidate(self, cid: str, cand_id: str, actor: str = "agent", **fields):
        """Write candidate fields. An outcome change goes through verdicts.family_change: later actors can only lower
        a verdict; a refused change is dropped, logged in the note, and its reason returned."""
        refused = None
        if "note" in fields and actor not in CODE_ACTORS:
            # the code-check tags freeze a verdict, so only code may write them: an agent quoting "[independent
            # check]" in its own note must not be able to lock a family (found in review)
            cur0 = self.one("SELECT note FROM candidates WHERE campaign_id=? AND id=?", (cid, cand_id)) or {}
            fields["note"] = _strip_code_tags(str(fields["note"] or ""), cur0.get("note") or "")
        if "outcome" in fields:
            from .verdicts import family_change
            cur = self.one("SELECT outcome, note FROM candidates WHERE campaign_id=? AND id=?", (cid, cand_id)) or {}
            ok, why = family_change(cur.get("outcome"), fields["outcome"], actor, cur.get("note") or "")
            if not ok:
                fields.pop("outcome")
                fields["note"] = (fields.get("note") or cur.get("note") or "") + f" [verdict rule: {why}]"
                refused = why
        if not fields:
            return refused
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        self.x(f"UPDATE candidates SET {cols} WHERE campaign_id=? AND id=?", (*vals, cid, cand_id))
        return refused

    # -- hypothesis board (the PI's research programme) ---------------------------------
    def add_hypothesis(self, cid: str, h: dict) -> str:
        hid = h.get("id") or new_id("H")
        now = time.time()
        conf = h.get("confirm", "")
        self.x("INSERT OR REPLACE INTO hypotheses(id,campaign_id,question,title,statement,test,kill,data,priority,origin,"
               "status,result,evidence,novelty,review,task_ids,created,updated,confirm,confirmation,novelty_basis) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (hid, cid, h.get("question", ""), h.get("title", "")[:200], h.get("statement", ""), h.get("test", ""),
                h.get("kill", ""), json.dumps(h.get("data", "")) if not isinstance(h.get("data"), str) else h.get("data"),
                float(h.get("priority", 0.5) or 0.5), h.get("origin", "pi"), h.get("status", "open"), "", "", "", "",
                "[]", now, now, json.dumps(conf) if not isinstance(conf, str) else conf, "",
                str(h.get("novelty_basis", ""))[:2000]))
        return hid

    def hypotheses(self, cid: str, **where) -> list[dict]:
        sql = "SELECT * FROM hypotheses WHERE campaign_id=?"
        args: list[Any] = [cid]
        for k, v in where.items():
            sql += f" AND {k}=?"
            args.append(v)
        rows = self.q(sql + " ORDER BY priority DESC, created", args)
        for r in rows:
            r["task_ids"] = json.loads(r["task_ids"] or "[]")
        return rows

    def update_hypothesis(self, cid: str, hid: str, actor: str = "agent", **fields):
        """Write hypothesis fields. A status change goes through verdicts.hypothesis_change (only down, except the
        code gates); a refused change is dropped and recorded in `evidence`."""
        refused = None
        if "status" in fields:
            from .verdicts import hypothesis_change
            cur = self.one("SELECT status, evidence FROM hypotheses WHERE campaign_id=? AND id=?", (cid, hid)) or {}
            ok, why = hypothesis_change(cur.get("status"), fields["status"], actor)
            if not ok:
                fields.pop("status")
                fields["evidence"] = (fields.get("evidence") or cur.get("evidence") or "") + f"\n[verdict rule: {why}]"
                refused = why
        if "task_ids" in fields and isinstance(fields["task_ids"], list):
            fields["task_ids"] = json.dumps(fields["task_ids"])
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        self.x(f"UPDATE hypotheses SET {cols} WHERE campaign_id=? AND id=?", (*vals, cid, hid))
        return refused

    # -- reports -----------------------------------------------------------------
    def add_report(self, cid: str, tid: str, family: str, title: str, body: str, figures: list[str]) -> str:
        rid = new_id("r_")
        self.x("INSERT INTO reports VALUES(?,?,?,?,?,?,?,?,?)",
               (rid, cid, tid, family, title, body, json.dumps(figures), "filed", time.time()))
        return rid

    # -- stats for the dashboard ---------------------------------------------------
    def stats(self, cid: str) -> dict:
        s = self.one("""SELECT COUNT(*) n, COALESCE(SUM(tokens_in),0) tin, COALESCE(SUM(tokens_out),0) tout,
                        COALESCE(SUM(COALESCE(ended, strftime('%s','now')) - started),0) secs
                        FROM sessions WHERE campaign_id=?""", (cid,)) or {}
        t = self.q("SELECT status, origin, COUNT(*) n FROM tasks WHERE campaign_id=? GROUP BY status, origin", (cid,))
        return {"sessions": s.get("n", 0), "tokens_in": s.get("tin", 0), "tokens_out": s.get("tout", 0),
                "agent_hours": round((s.get("secs") or 0) / 3600, 2), "tasks": t}
