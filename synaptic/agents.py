"""Agent sessions. One WorkerSession per task (kept alive across revisions, like the paper's
worker/supervisor pair), plus single-shot calls for planner, supervisor, curator, triage, editor, judge."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import prompts
from .config import HARNESS, MODELS, SKILLS_DIR
from .llm import image_part
from .protocol import parse_action, parse_json
from .record import Record
from .tools import ToolContext, toolbox


def skills_index() -> str:
    out = []
    for p in sorted(SKILLS_DIR.glob("*.md")):
        first = next((l.strip("# ").strip() for l in p.read_text().splitlines() if l.strip()), p.stem)
        out.append(f"  - {p.stem}: {first}")
    return "\n".join(out) or "  (none)"


def _short(v, n=160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[:n] + "..."


class Runner:
    """Shared plumbing: LLM calls with accounting and event logging."""

    def __init__(self, record: Record, llm, campaign_id: str, campaign_dir: Path, mock: bool):
        self.record, self.llm, self.cid, self.cdir, self.mock = record, llm, campaign_id, campaign_dir, mock

    ACTION_SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}, "args": {"type": "object"}},
                     "required": ["action", "args"]}

    def call(self, role: str, messages: list[dict], task_id: str | None, session_id: str,
             max_tokens: int | None = None):
        kw = {}
        if role == "worker" and getattr(MODELS, "structured", False) and not self.mock:
            kw["schema"] = self.ACTION_SCHEMA
        c = self.llm.complete(messages, role, max_tokens=max_tokens, **kw)
        self.record.add_usage(session_id, c.tokens_in, c.tokens_out)
        return c

    def oneshot(self, role: str, system: str, user: str | list, task_id: str | None = None) -> dict:
        sid = self.record.open_session(self.cid, task_id, role, MODELS.model_for(role) if not self.mock else "mock")
        try:
            base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            msgs = base
            # found live: the PI's 24k-character programme failed to parse 3 times in a row, while the same call made
            # fresh parsed at once. Pasting a long broken answer back makes the next one worse, so retry fresh first,
            # then once with a repair request, then fresh again; keep the unparsed text for diagnosis.
            for attempt in range(4):
                c = self.call(role, msgs, task_id, sid)
                if c.reasoning:
                    self.record.event(self.cid, sid, task_id, role, "thought", c.reasoning[-6000:])
                obj = parse_json(c.text)
                if obj is not None:
                    self.record.event(self.cid, sid, task_id, role, "verdict", json.dumps(obj)[:8000])
                    return obj
                self.record.event(self.cid, sid, task_id, role, "unparsed",
                                  f"finish={c.finish_reason} chars={len(c.text)}\n{c.text[:3000]}\n...\n{c.text[-3000:]}")
                msgs = base if attempt != 1 else base + [
                    {"role": "assistant", "content": c.text},
                    {"role": "user", "content": "That answer was not valid JSON. Answer with exactly one JSON block as "
                                                "specified: double quotes, no comments, no trailing commas."}]
            self.record.event(self.cid, sid, task_id, role, "error", "no JSON after 4 attempts")
            return {}
        finally:
            self.record.close_session(sid)

    # -- roles -------------------------------------------------------------------
    def supervise_plan(self, campaign: dict, task: dict, plan: dict) -> dict:
        user = (f"Campaign: {campaign['brief'].get('title')}\nTask ({task['kind']}): {task['title']}\n"
                f"{task['brief']}\n<task_meta>{json.dumps(task['meta'], default=str)}</task_meta>\n\n"
                f"Worker plan:\n{json.dumps(plan, indent=1)}")
        return self.oneshot("supervisor", prompts.SUPERVISOR_PLAN, user, task["id"]) or {"decision": "approve"}

    def supervise_result(self, campaign: dict, task: dict, result: dict, digest: str, figures: list[str],
                         revision: int = 0, previous_feedback: str = "") -> dict:
        rnote = ""
        if revision:
            rnote = (f"\n\n## This is revision {revision}. Your previous feedback was:\n{previous_feedback}\n"
                     "Judge ONLY whether those points were addressed. Do not add new requirements now: if the earlier "
                     "points are addressed, ACCEPT and list any remaining limitation as a caveat in your feedback (the "
                     "curator files caveats). Request another revision only if a previous point is still unaddressed or a "
                     "claim is contradicted by the trace.")
        text = (f"Campaign: {campaign['brief'].get('title')}\nTask ({task['kind']}): {task['title']}\n{task['brief']}\n"
                f"<task_meta>{json.dumps(task['meta'], default=str)}</task_meta>\n\n"
                f"## Worker summary\n{result.get('summary', '')}\n\n## Verdict recorded\n{result.get('verdict')}\n\n"
                f"## Action digest\n{digest}\n\n## Figures attached: {', '.join(Path(f).name for f in figures) or 'none'}"
                + rnote)
        content: list | str = text
        if figures and MODELS.vision:
            content = [{"type": "text", "text": text}] + [image_part(f) for f in figures[:3] if Path(f).exists()]
        out = self.oneshot("supervisor", prompts.SUPERVISOR_RESULT, content, task["id"])
        out.setdefault("decision", "accept")
        out.setdefault("followups", [])
        return out

    def curate(self, task: dict, result: dict, review: dict):
        user = (f"Task ({task['kind']}): {task['title']}\nBrief: {task['brief']}\n\nWorker summary:\n"
                f"{result.get('summary', '')}\n\nVerdict: {result.get('verdict')}\n"
                f"Supervisor: {review.get('decision')} - {review.get('feedback', '')}")
        out = self.oneshot("curator", prompts.CURATOR, user, task["id"])
        for e in (out.get("entries") or [])[:6]:
            self.record.kb_add(self.cid, task["id"], e.get("kind", "finding"), _short(e.get("title", ""), 200),
                               e.get("body", ""), e.get("tags") or [])

    def plan_campaign(self, brief: dict) -> dict:
        return self.oneshot("planner", prompts.PLANNER, "Research brief:\n" + json.dumps(brief, indent=1))

    def triage(self, proposals: list[dict], state: str) -> list[dict]:
        listing = "\n".join(f"[{i}] (from {p['origin']}, parent '{p['parent_title']}') {p['title']}\n    {p['brief']}"
                            f"\n    rationale: {p.get('rationale', '')}" for i, p in enumerate(proposals))
        out = self.oneshot("triage", prompts.TRIAGE, f"Campaign state:\n{state}\n\nProposals:\n{listing}")
        return out.get("decisions") or [{"index": i, "decision": "release", "reason": "default"} for i in range(len(proposals))]

    def edit(self, report_md: str, task_id: str) -> dict:
        return self.oneshot("editor", prompts.EDITOR, report_md, task_id) or {"decision": "accept"}

    def edit_fix(self, report_md: str, feedback: str, task_id: str) -> str:
        """After two rounds of revision requests the editor fixes the wording itself instead of sending the report
        back again (v3: reports stalled after three revisions, mostly over wording). Numbers are never invented:
        what the feedback asks that the report cannot support becomes an explicit limitation."""
        out = self.oneshot("editor", prompts.EDITOR_FIX, f"# Reviewer feedback\n{feedback}\n\n# Report\n{report_md}",
                           task_id)
        return (out or {}).get("report") or report_md

    def audit(self, family: dict, note: str, summary: str, task_id: str | None) -> dict:
        user = (f"Family: {json.dumps(family, default=str)}\n\nVerdict note:\n{note}\n\n"
                f"Deep-dive report:\n{summary[:14000]}")
        return self.oneshot("auditor", prompts.AUDITOR, user, task_id) or {}

    def pi_program(self, context: str) -> dict:
        return self.oneshot("pi", prompts.PI_PROGRAM, context)

    def pi_confirm_fix(self, context: str) -> dict:
        return self.oneshot("pi", prompts.PI_CONFIRM_FIX, context)

    def pi_meeting(self, context: str) -> dict:
        return self.oneshot("pi", prompts.PI_MEETING, context)

    def judge(self, a: str, b: str) -> dict:
        return self.oneshot("judge", prompts.JUDGE, f"## Report A\n{a}\n\n## Report B\n{b}")


class WorkerSession:
    def __init__(self, runner: Runner, campaign: dict, task: dict, data_index: dict):
        self.r = runner
        self.campaign = campaign
        self.task = task
        kind = task["kind"]
        self.tools = toolbox("science" if task.get("stage") == "science" else
                             ("referee" if kind == "referee" else "hypothesis") if task.get("stage") == "program" else kind)
        if not (task.get("meta") or {}).get("family"):
            self.tools.pop("set_verdict", None)  # survey-stage follow-ups have no family yet: conclusions go in the summary
        wd = runner.cdir / "tasks" / task["id"]
        wd.mkdir(parents=True, exist_ok=True)
        try:  # shared downloads appear as ./data/ inside every task folder
            (runner.cdir / "data").mkdir(exist_ok=True)
            if not (wd / "data").exists():
                (wd / "data").symlink_to(runner.cdir / "data", target_is_directory=True)
        except OSError:
            pass
        self.sid = runner.record.open_session(runner.cid, task["id"], "worker",
                                              MODELS.model_for("worker") if not runner.mock else "mock")
        self.ctx = ToolContext(runner.record, runner.cid, task["id"], self.sid, wd, runner.cdir, mock=runner.mock)
        self.ctx.state.update({"data_files": {}, "followups": [], "trace": []})
        th = runner.record.thresholds(runner.cid)
        kb = runner.record.kb_search(runner.cid, f"{task['title']} {json.dumps(task['meta'], default=str)}", 6)
        kb_digest = "\n".join(f"- [{k['kind']}] {k['title']}: {_short(k['body'], 400)}" for k in kb)
        tools_doc = "\n".join(t.doc() for t in self.tools.values())
        import os

        ed = bool(os.getenv('EARTHDATA_TOKEN'))
        avail = [f"merra2_series: {'AVAILABLE (Earthdata token configured): use it' if ed else 'not available in this run (no Earthdata token)'}",
                 f"nasa_series (GPM IMERG rain, GLDAS soil moisture/evaporation): {'AVAILABLE: use it instead of saying IMERG or GLDAS cannot be downloaded' if ed else 'not available (no Earthdata token)'}",
                 f"gpu_submit: {'Kaggle GPU (2x T4) configured' if os.getenv('KAGGLE_API_TOKEN') else 'no Kaggle account: runs on local CPU'}"
                 " -- templates teleconnection / causal / predictability prepare the campaign data themselves"]
        trusted = {p.stem.replace("trusted_", ""): f"data/{p.name}" for p in sorted((runner.cdir / "data").glob("trusted_*.csv"))}
        if trusted:
            tools_doc += ("\nTRUSTED TABLES (same format as the raw tables; values the data screen flagged are removed, "
                          "and for a cell with a break only the longer homogeneous segment is kept). Analyse these by "
                          "default instead of filtering trust.csv yourself: " + json.dumps(trusted))
        tools_doc += ("\nService status for this run (CURRENT: it overrides any older knowledge-base note that says a "
                      "dataset or tool is unavailable):\n- " + "\n- ".join(avail))
        self.messages = [{"role": "system", "content": prompts.worker_system(
            campaign, {**task, "workdir": str(wd)}, th, tools_doc, kb_digest, skills_index(), data_index)},
            {"role": "user", "content": "Begin. Your first action must be `plan`."}]
        self.phase = "plan"
        self.plan_rounds = 0
        self.steps = 0
        self.parse_failures = 0

    # -- event helpers -------------------------------------------------------------
    def ev(self, type_: str, content: str, meta: dict | None = None, role: str = "worker"):
        self.r.record.event(self.r.cid, self.sid, self.task["id"], role, type_, content, meta)

    def _prune(self) -> list[dict]:
        msgs = []
        keep_img = 2
        n = len(self.messages)
        for i in range(n - 1, -1, -1):
            m = self.messages[i]
            c = m["content"]
            if isinstance(c, list):
                has_img = any(p.get("type") == "image_url" for p in c)
                if has_img and keep_img > 0:
                    keep_img -= 1
                elif has_img:
                    c = [p for p in c if p.get("type") == "text"]
                    c = [{"type": "text", "text": c[0]["text"] + "\n[earlier image removed to save context]"}]
            if m["role"] in ("user", "assistant") and i < n - 10 and i > 1:
                txt = c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
                if len(txt) > 1500 and m["role"] == "user":
                    c = txt[:700] + "\n...[older observation truncated]...\n" + txt[-300:]
                elif len(txt) > 1500:  # keep which action was taken, drop the long script/report it carried
                    act, _, _ = parse_action(txt)
                    name = (act or {}).get("action", "unknown")
                    c = (txt.split("```")[0][:500] + "\n...[older reply truncated]...\n```json\n" +
                         json.dumps({"action": name, "args": {"_truncated": True}}) + "\n```")
            msgs.append({**m, "content": c})
        msgs = list(reversed(msgs))
        # hard context budget: long revision cycles once grew a session past the model's 1M-token window
        budget = 700_000  # characters, roughly 200k tokens
        def size(ms):
            return sum(len(x["content"]) if isinstance(x["content"], str) else
                       sum(len(q.get("text", "")) for q in x["content"]) for x in ms)
        if size(msgs) > budget:
            for j in range(2, max(2, len(msgs) - 6)):
                c = msgs[j]["content"]
                txt = c if isinstance(c, str) else "\n".join(q.get("text", "") for q in c if q.get("type") == "text")
                if len(txt) > 400:
                    msgs[j] = {**msgs[j], "content": txt[:250] + "\n...[trimmed to fit context]...\n" + txt[-100:]}
                if size(msgs) <= budget:
                    break
        return msgs

    def _observe(self, text: str, images: list[str] | None = None):
        if images and MODELS.vision:
            content = [{"type": "text", "text": f"Observation:\n{text}"}] + [image_part(p) for p in images]
        else:
            content = f"Observation:\n{text}"
            if images:
                content += "\n(vision disabled: the image was not shown; inspect numbers with python/look instead)"
        self.messages.append({"role": "user", "content": content})

    # -- gates ---------------------------------------------------------------------
    def _finish_blockers(self) -> list[str]:
        s, k = self.ctx.state, self.task["kind"]
        out = []
        if HARNESS.require_look and not s.get("looked"):
            out.append("You never used `look`. Inspect the raw data and your key figure(s) directly before finishing.")
        if k == "sweep" and not s.get("registered"):
            out.append("Register your sweep table with register_sweep before finishing.")
        if k == "classify" and not s.get("families_built"):
            out.append("Call build_families before finishing.")
        if k == "census" and not s.get("promote_called") and not s.get("none_passed"):
            out.append("Call apply_filters, then promote the families you select (or explain none passed).")
        if k == "science" and not s.get("findings"):
            out.append("Record your scientific result(s) with record_finding before finishing (a clearly stated "
                       "negative result counts).")
        if k == "hypothesis" and not s.get("hypothesis_result"):
            out.append(f"Close the test with record_hypothesis_result for {self.task['meta'].get('hypothesis')} "
                       "(not_shown, refuted, known and inconclusive are valid results; not significant = not_shown).")
        if k == "referee" and not s.get("referee"):
            out.append("File your decision with referee_report (upheld, downgraded, not_shown or refuted) before finishing.")
        if k == "deep_dive" and self.task["meta"].get("family") and not s.get("verdict"):
            out.append(f"Record your verdict on {self.task['meta']['family']} with set_verdict before finishing.")
        return out

    # -- main loop -----------------------------------------------------------------
    def run(self, feedback: str | None = None) -> dict:
        if feedback:
            self.phase = "act"
            self._observe(f"SUPERVISOR REVIEW - revise your work:\n{feedback}\nAddress every point, then finish again.")
        budget = HARNESS.max_steps + (6 if feedback else 0)
        local_steps = 0
        while True:
            local_steps += 1
            if local_steps > budget + 2:
                self.ev("error", "step budget exhausted without finish")
                return {"status": "stalled", "summary": "Step budget exhausted before the worker finished.",
                        "figures": [], "verdict": self.ctx.state.get("verdict")}
            if local_steps == budget - 3:
                todo = self._finish_blockers()
                self.messages.append({"role": "user", "content": "Note: 3 actions left in your budget. Wrap up." + (
                    " Before finish is accepted you still need: " + " ".join(todo) if todo else "")})
            if local_steps == budget:
                self.messages.append({"role": "user", "content": "Budget reached: call finish NOW with what you have."})
            c = self.r.call("worker", self._prune(), self.task["id"], self.sid)
            action, thought, err = parse_action(c.text)
            if err and c.reasoning:  # some reasoning models leave the final action inside the reasoning channel
                a2, _, err2 = parse_action(c.reasoning.split("</think>")[-1][-6000:])
                if not err2:
                    action, err = a2, None
            if err and c.finish_reason == "length":
                err = ("Your reply was cut off by the output token limit before the action was complete. Think "
                       "more briefly, keep scripts short, and split big analyses across several python calls.")
            if c.reasoning or thought:
                self.ev("thought", ((c.reasoning or "") + ("\n\n" if c.reasoning and thought else "") + (thought or ""))[-8000:])
            self.messages.append({"role": "assistant", "content": c.text})
            if err:
                self.parse_failures += 1
                self.ev("error", err, {"raw_head": c.text[:1500], "raw_tail": c.text[-1500:],
                                       "finish_reason": c.finish_reason, "reasoning_tail": (c.reasoning or "")[-800:]})
                if self.parse_failures > 5:
                    return {"status": "failed", "summary": "Worker could not follow the action format.", "figures": []}
                self._observe(err)
                continue
            name, args = action["action"], action.get("args", {})
            self.ev("action", f"{name}({_short(args, 1500)})", {"name": name, "args": args})
            self.ctx.state["trace"].append(name)
            if self.phase == "plan" and name != "plan":
                self._observe("Submit your plan first: {\"action\": \"plan\", \"args\": {\"steps\": [...], \"rationale\": \"...\"}}")
                continue
            if name == "plan":
                self.plan_rounds += 1
                self.ctx.state["plan"] = args
                rev = self.r.supervise_plan(self.campaign, self.task, args)
                self.ev("plan_review", f"{rev.get('decision')}: {rev.get('feedback', '')}", rev, role="supervisor")
                if rev.get("decision") == "revise" and self.plan_rounds < 3:
                    self._observe(f"Supervisor asks you to revise the plan: {rev.get('feedback', '')}")
                else:
                    self.phase = "act"
                    self._observe(f"Plan approved. Supervisor note: {rev.get('feedback', '') or 'none'}. Proceed.")
                continue
            if name == "propose_followup":
                if len(self.ctx.state["followups"]) >= HARNESS.max_followups_per_task:
                    self._observe("Follow-up limit for this task reached.")
                else:
                    self.ctx.state["followups"].append(args)
                    self._observe("Follow-up proposal queued for triage. Continue your task.")
                continue
            if name == "read_skill":
                p = SKILLS_DIR / f"{str(args.get('name', '')).replace('/', '')}.md"
                self._observe(p.read_text() if p.exists() else f"No skill '{args.get('name')}'.")
                continue
            if name == "finish":
                summary = str(args.get("summary") or args.get("report") or args.get("result") or "").strip()
                if len(summary) < 20 and thought and len(thought.strip()) >= 20:
                    summary = thought.strip()  # Nemotron often writes the report as prose next to an empty finish
                blockers = self._finish_blockers()
                if len(summary) < 20:
                    blockers.append("Your finish has no report. Put the full report in args.summary: what you did, the key "
                                    "numbers, what you looked at and what you saw, anything odd, and your conclusion.")
                if blockers and local_steps >= budget and len(summary) >= 200:
                    # out of steps with a real report: keep the work, and make the unmet checks visible to the supervisor
                    summary += "\n\n[Finished at the step limit. Checks NOT completed: " + " ".join(blockers) + "]"
                    args["summary"] = summary
                    self.ev("gate", "accepted at step limit with unmet checks: " + " | ".join(blockers), role="harness")
                    blockers = []
                if blockers:
                    self.ev("gate", " | ".join(blockers), role="harness")
                    self._observe("Finish refused:\n- " + "\n- ".join(blockers))
                    continue
                figs = []
                for f in args.get("figures") or []:
                    try:
                        p = self.ctx.resolve(f)
                        if p.exists():
                            figs.append(str(p))
                    except Exception:  # noqa: BLE001
                        pass
                return {"status": "finished", "summary": summary, "figures": figs,
                        "confidence": args.get("confidence"), "verdict": self.ctx.state.get("verdict"),
                        "followups": list(self.ctx.state["followups"])}
            tool = self.tools.get(name)
            if not tool:
                self._observe(f"Unknown action '{name}'. Available: {', '.join(self.tools)} + plan, propose_followup, "
                              f"read_skill, finish.")
                continue
            res = tool.run(self.ctx, args)
            self.steps += 1
            self._after_tool(name, args, res)
            if res.data_chars:
                self.r.record.add_data_chars(self.sid, res.data_chars)
            self.ev("observation", res.text[:6000], {"tool": name, "error": res.is_error, "files": res.files[:20]})
            for img in res.images:
                self.ev("image", img, {"tool": name})
            self._observe(res.text, res.images)

    def _after_tool(self, name: str, args: dict, res):
        s = self.ctx.state
        if res.is_error:
            return
        s.setdefault("ok_tools", []).append(name)
        if name == "power_region" and res.files:
            for v in args.get("parameters") if isinstance(args.get("parameters"), list) else \
                    str(args.get("parameters", "")).replace(",", " ").split():
                s["data_files"][v.upper()] = res.files[0]
        if name == "register_sweep":
            s["registered"] = True
        if name == "build_families":
            s["families_built"] = True
        if name == "apply_filters":
            s["filters_applied"] = True
            lines = res.text.split("\n")
            if len(lines) > 1 and lines[1].startswith("0 of"):
                s["none_passed"] = True
        if name == "promote":
            s["promote_called"] = True

    def digest(self) -> str:
        """Action trace for the supervisor. Observations are EXCERPTS (marked as such) so the reviewer does not
        mistake a digest cut for a truncated analysis."""
        rows = self.r.record.q("SELECT type, content, meta FROM events WHERE session_id=? AND type IN "
                               "('action','observation','gate','image') ORDER BY id", (self.sid,))
        out = [f"(Excerpts only: each observation below is cut to a few hundred characters for this digest; the worker "
               f"saw the full outputs. Raw data characters read by the worker in total: "
               f"{self.ctx.state.get('data_chars', 0)}; longest contiguous raw block: {self.ctx.state.get('longest_run', 0)}.)"]
        for e in rows[-70:]:
            if e["type"] == "action":
                out.append(f"> {e['content'][:300]}")
            elif e["type"] == "gate":
                out.append(f"  [finish refused] {e['content'][:200]}")
            elif e["type"] == "image":
                out.append(f"  [worker looked at image {Path(e['content']).name}]")
            else:
                tool = json.loads(e["meta"] or "{}").get("tool", "")
                lim = 900 if tool in ("look", "python") else 350
                txt = e["content"]
                more = f" ...[excerpt; {len(txt)} chars in full]" if len(txt) > lim else ""
                out.append(f"  = {txt[:lim]}{more}")
        return "\n".join(out)

    def close(self):
        self.r.record.close_session(self.sid)
