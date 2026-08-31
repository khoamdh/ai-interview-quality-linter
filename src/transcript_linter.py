#!/usr/bin/env python3

"""
Northwind Transcript Linter
===========================

Purpose
-------
This script evaluates the 600 fictional Northwind employee interview
transcripts using a set of deterministic and heuristic quality checks.

The analysis focuses on whether the AI interviewer:
- detects and develops meaningful employee signals,
- avoids repeated or rephrased questions,
- avoids leading questions and assumed emotions,
- stops appropriately when little new information is available,
- avoids unproductive pressing loops, and
- potentially misses opportunities to develop useful employee signals.

Each conversation is assigned one of five screening outcomes:
1. INSIGHTFUL_CLEAN
2. INSIGHTFUL_WITH_DEFECTS
3. NO_INSIGHT_MISSED_SIGNAL
4. NO_INSIGHT_AI_LOOP
5. NO_INSIGHT_NO_SIGNAL

Important
---------
The results are deterministic screening outputs, not human-validated
ground truth. In particular, classifications such as "usable insight"
and "missed signal" rely on heuristic rules and should be validated
against a manually reviewed sample.

Usage
-----
Place conversations.json in the same directory as this script and run:

    python northwind_transcript_linter.py

Or specify an input file:

    python northwind_transcript_linter.py path/to/conversations.json

Optional output arguments:

    python northwind_transcript_linter.py conversations.json \
        --out-csv northwind_linter_results.csv \
        --out-json northwind_linter_results.json \
        --summary northwind_linter_summary.json

Outputs
-------
northwind_linter_results.csv
    Conversation-level results in tabular format.

northwind_linter_results.json
    Conversation-level results in JSON format.

northwind_linter_summary.json
    Aggregate outcome counts, defect counts, and example conversation IDs.

Requirements
------------
Python 3.x. The script uses Python standard-library modules only.
"""

import argparse, csv, json, re
from difflib import SequenceMatcher
from pathlib import Path

LEADING = [
    r"\bor .*business as usual\b",
    r"\bor is it too early to say\b",
    r"\bor has everything been fine\b",
]
NOTHING = [
    r"nothing('s| has) really changed", r"nothing has changed",
    r"same as before", r"business as usual", r"not really, no",
    r"same answer really", r"like i said, the same as before",
    r"ennallaan on jatkunut", r"sama juttu kuin ennen",
    r"ei ole oikeastaan muuttunut mikään",
]
LOW_INFO = NOTHING + [
    r"\bi don't know\b", r"\bi wouldn't know\b", r"\bi couldn't say\b",
    r"\byou'd have to ask someone else\b", r"\bit's fine\b",
    r"\bit's been alright i suppose\b", r"\ben osaa sanoa\b",
    r"\ben tiedä\b", r"\bihan hyvin\b",
]
OUT_OF_SIGHT = [
    r"\bi('m| am) in the warehouse\b", r"\bi don't deal with that\b",
    r"\bthat's not something i see\b", r"\byou'd have to ask someone else\b",
    r"\bi wouldn't know\b", r"\bi couldn't say\b",
]
SIGNAL_TERMS = [
    "customer","supplier","register","system","software","licence","license",
    "price","pricing","discount","rota","schedule","training","equipment",
    "machine","line","lease","office move","job","role","approval","spending",
    "owner","ownership","ticketing","depot","site","process","delivery","orders",
    "quote","quoting","pay","holiday","maintenance","recalibrating","kit",
    "asiakas","toimittaja","rekisteri","järjestelmä","hinnoittelu","alennus",
    "vuorolista","vuorojärjestely","koulutus","laite","linja","rooli","prosessi",
]
IMPACT_TERMS = [
    "because","which means","means that","caused","causes","late","delay","slower",
    "can't","cannot","unable","without knowing","confusing","unclear","mess",
    "haywire","tense","lost","copying","by hand","driving between","pushed back",
    "queried","inconsistent","getting in the way",
    "koska","joten","tarkoittaa","myöhässä","viivästyy","epäselvä","sekava","käsin",
]
ACTION_TERMS = [
    "would help","fix first","need to hear from","clarified","clarify",
    "who would need","who needs","who owns","replace","recalibrate","answer first",
    "auttaisi","korjata","selventää","kuka voisi ratkaista","kuka omistaa",
]
GENERIC_Q = [
    r"^what has been on your mind", r"^how have things been",
    r"^how have the first few weeks felt", r"^what was your first thought",
    r"^what has got in the way", r"^what would help you most",
    r"^what would you want answered first", r"^what are you least sure about",
    r"^anything we have not covered", r"^is there anything else",
    r"^what have i not asked about", r"^what has gone better than you expected",
    r"^what is the one thing you would fix first",
    r"^miltä ensimmäiset viikot", r"^miten arki on sujunut",
    r"^mikä oli ensimmäinen ajatuksesi", r"^onko vielä jotain", r"^mikä auttaisi",
]
PARAPHRASE = [r"^so what you're saying is\b", r"^eli tarkoitat"]
ASSUMED_EMOTION = [
    r"\bi can hear that this has been frustrating\b",
    r"\bit sounds like this has been quite an uncertain time\b",
    r"\byou('re| are) (worried|excited|frustrated|apprehensive|uncertain)\b",
]

STOP = set("""the a an and or but to of for in on at as is are was were be been being has have had
do does did what how why when where who which would could should you your me my we our i it this
that those these from with about into than then so very just really mostly any anything something
else most more since merger announcement first few weeks ja tai mutta että se on ovat oli olivat
olla ollut mitä miten miksi milloin missä kuka mikä voisi pitäisi sinä sinun minä me meidän tämä tuo
nämä lisää vielä jotain nyt yhdistymisestä ilmoituksen jälkeen""".split())

def norm(s):
    s=(s or "").lower().replace("’","'").strip()
    return re.sub(r"\s+"," ",s)

def exact_norm(s):
    return re.sub(r"\s+"," ",re.sub(r"[^\wåäö0-9']+"," ",norm(s))).strip()

def wc(s): return len(re.findall(r"\b[\wåäö]+\b",s or "",re.UNICODE))
def match_any(s, pats): return any(re.search(p,norm(s)) for p in pats)
def has_term(s, terms):
    t=norm(s)
    return any(x in t for x in terms)

def toks(s):
    return {x for x in re.findall(r"[a-zåäö0-9]+",norm(s)) if x not in STOP and len(x)>2}

def sim(a,b):
    seq=SequenceMatcher(None,norm(a),norm(b)).ratio()
    ta,tb=toks(a),toks(b)
    jac=len(ta&tb)/len(ta|tb) if ta or tb else 0
    return .6*seq+.4*jac

def low_info(s):
    return wc(s)<=3 or match_any(s,LOW_INFO)

def signal(s):
    if low_info(s): return False
    return has_term(s,SIGNAL_TERMS) or (wc(s)>=16 and (has_term(s,IMPACT_TERMS) or has_term(s,ACTION_TERMS)))

def insight_strength(s):
    if not signal(s): return 0
    return 2 if has_term(s,IMPACT_TERMS) or has_term(s,ACTION_TERMS) else 1

def generic_q(s): return match_any(s,GENERIC_Q)
def paraphrase(s): return match_any(s,PARAPHRASE)

def builds_on(q,e):
    if paraphrase(q): return False
    if toks(q)&toks(e): return True
    probes=[
        r"\bcan you give me an example\b",r"\bwhat happened\b",r"\bwhat difference has it made\b",
        r"\bhow has that affected\b",r"\bwhat does that mean for\b",r"\bwho (puts|owns|decides|signs|needs)\b",
        r"\bwhat are you doing in the meantime\b",r"\bwho would need to make that happen\b",
        r"\bwhen did\b",r"\bhow often\b",r"\bmiten se on vaikuttanut\b",r"\bkuka .* omistaa\b",
    ]
    return match_any(q,probes)

def repeat_type(a,b):
    if wc(a)<4 or wc(b)<4:return "none"
    if exact_norm(a)==exact_norm(b):return "exact"
    return "near" if sim(a,b)>=.86 else "none"

def lint(c):
    turns=c.get("turns",[])
    hard=[]; soft=[]
    ai=[t for t in turns if t.get("role")=="ai"]
    qs=[(t.get("i"),t.get("text","")) for t in ai if "?" in t.get("text","")]

    for pos,t in enumerate(turns):
        if t.get("role")!="ai": continue
        txt=t.get("text","")
        if wc(txt)>20: soft.append(("question_over_20_words",t.get("i")))
        if txt.count("?")>=2: hard.append(("multiple_questions",t.get("i")))
        if match_any(txt,LEADING): hard.append(("leading_question",t.get("i")))
        if paraphrase(txt): hard.append(("broken_or_shallow_paraphrase",t.get("i")))
        if match_any(txt,ASSUMED_EMOTION): hard.append(("assumes_emotion",t.get("i")))

    for a in range(len(qs)):
        for b in range(a+1,len(qs)):
            ia,qa=qs[a]; ib,qb=qs[b]
            if ib-ia>8: continue
            s=sim(qa,qb)
            if s>=.82: hard.append(("repeated_or_rephrased_question",ib))
            elif s>=.68: soft.append(("possible_rephrased_question",ib))

    for j,t in enumerate(turns):
        if t.get("role")=="employee" and match_any(t.get("text",""),NOTHING):
            if j+1<len(turns) and turns[j+1].get("role")=="ai":
                q=norm(turns[j+1].get("text",""))
                if any(x in q for x in ["what is different","what has changed","why it feels the same",
                                        "what difference has","mikä on työssäsi toisin","mikä on muuttunut"]):
                    hard.append(("reprobes_nothing_changed",turns[j+1].get("i")))

    for j,t in enumerate(turns):
        if t.get("role")=="employee" and match_any(t.get("text",""),OUT_OF_SIGHT):
            if j+1<len(turns) and turns[j+1].get("role")=="ai" and j>0:
                if sim(turns[j-1].get("text",""),turns[j+1].get("text",""))>=.65:
                    hard.append(("persists_after_out_of_sight_answer",turns[j+1].get("i")))

    signal_pos=[]; strong_pos=[]; followed=0; abandoned=0; missed=[]
    for j,t in enumerate(turns):
        if t.get("role")!="employee": continue
        st=insight_strength(t.get("text",""))
        if st>=1: signal_pos.append(j)
        if st>=2: strong_pos.append(j)
        if st>=1:
            if j+1<len(turns) and turns[j+1].get("role")=="ai":
                q=turns[j+1].get("text","")
                if builds_on(q,t.get("text","")): followed+=1
                else:
                    abandoned+=1; missed.append(turns[j+1].get("i"))
            elif j==len(turns)-1:
                abandoned+=1

    signal_present=bool(signal_pos)
    usable=bool(strong_pos) or (signal_present and followed>0)

    exact=near=forced=0
    forced_examples=[]
    emp_idx=[i for i,t in enumerate(turns) if t.get("role")=="employee"]
    for a in range(len(emp_idx)):
        ia=emp_idx[a]
        for b in range(a+1,len(emp_idx)):
            ib=emp_idx[b]
            if ib-ia>6: break
            rt=repeat_type(turns[ia].get("text",""),turns[ib].get("text",""))
            if rt=="none": continue
            if rt=="exact": exact+=1
            else: near+=1
            if ib==ia+2 and turns[ia+1].get("role")=="ai":
                q=turns[ia+1].get("text","")
                if (not builds_on(q,turns[ia].get("text",""))) or paraphrase(q) or generic_q(q):
                    forced+=1
                    forced_examples.append((turns[ia].get("i"),turns[ia+1].get("i"),turns[ib].get("i"),rt))

    loop_evidence=[]
    low_emp=[i for i,t in enumerate(turns) if t.get("role")=="employee" and low_info(t.get("text",""))]
    for k in range(len(low_emp)-1):
        a,b=low_emp[k],low_emp[k+1]
        if b==a+2 and turns[a+1].get("role")=="ai":
            q=turns[a+1].get("text","")
            if generic_q(q) or paraphrase(q) or match_any(turns[a].get("text",""),NOTHING):
                loop_evidence.append(turns[a+1].get("i"))
    loop=len(loop_evidence)>=2

    zero=not usable
    no_signal=zero and not signal_present
    missed_signal=zero and signal_present and (abandoned>0 or forced>0)
    ai_loop=zero and loop
    clear_defect=bool(hard)

    if usable and not clear_defect: outcome="INSIGHTFUL_CLEAN"
    elif usable and clear_defect: outcome="INSIGHTFUL_WITH_DEFECTS"
    elif missed_signal: outcome="NO_INSIGHT_MISSED_SIGNAL"
    elif ai_loop: outcome="NO_INSIGHT_AI_LOOP"
    else: outcome="NO_INSIGHT_NO_SIGNAL"

    return {
        "id":c.get("id"),"department":c.get("department"),"legacy_company":c.get("legacy_company"),
        "language":c.get("language"),"temperature":c.get("temperature"),"prompt_version":c.get("prompt_version"),
        "n_turns":len(turns),"conversation_outcome":outcome,
        "usable_insight_present":usable,"zero_usable_insight":zero,"signal_ever_present":signal_present,
        "zero_insight_no_signal_observed":no_signal,"zero_insight_interviewer_failure":missed_signal,
        "zero_insight_ai_loop":ai_loop,"unproductive_pressing_loop":loop,
        "clear_interviewing_defect":clear_defect,"hard_defect_count":len(hard),"soft_flag_count":len(soft),
        "hard_defect_types":sorted(set(x[0] for x in hard)),"soft_flag_types":sorted(set(x[0] for x in soft)),
        "employee_exact_repeat_count":exact,"employee_near_repeat_count":near,
        "forced_answer_repetition":forced>0,"forced_answer_repeat_count":forced,
        "substantive_signal_count":len(signal_pos),"signals_followed_up":followed,
        "signals_abandoned":abandoned,"missed_signal_count":len(missed),
        "forced_repeat_examples":forced_examples,"missed_signal_ai_turns":missed,"loop_ai_turns":loop_evidence
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="conversations.json")
    ap.add_argument("--out-csv", default="northwind_linter_results.csv")
    ap.add_argument("--out-json", default="northwind_linter_results.json")
    ap.add_argument("--summary", default="northwind_linter_summary.json")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = [lint(c) for c in data["conversations"]]

    fields = [
        "id","department","legacy_company","language","temperature","prompt_version","n_turns",
        "conversation_outcome","usable_insight_present","zero_usable_insight","signal_ever_present",
        "zero_insight_no_signal_observed","zero_insight_interviewer_failure","zero_insight_ai_loop",
        "unproductive_pressing_loop","clear_interviewing_defect","hard_defect_count","soft_flag_count",
        "hard_defect_types","soft_flag_types","employee_exact_repeat_count","employee_near_repeat_count",
        "forced_answer_repetition","forced_answer_repeat_count","substantive_signal_count",
        "signals_followed_up","signals_abandoned","missed_signal_count"
    ]

    # CSV output
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in rows:
            z = {k: r[k] for k in fields}
            z["hard_defect_types"] = ";".join(z["hard_defect_types"])
            z["soft_flag_types"] = ";".join(z["soft_flag_types"])
            w.writerow(z)

    # Full JSON output for all 600 conversations
    Path(args.out_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    outcomes = {}
    for r in rows:
        outcomes[r["conversation_outcome"]] = (
            outcomes.get(r["conversation_outcome"], 0) + 1
        )

    htypes = {}
    for r in rows:
        for x in r["hard_defect_types"]:
            htypes[x] = htypes.get(x, 0) + 1

    summary = {
        "n_conversations": len(rows),
        "outcome_counts": outcomes,
        "usable_insight_present": sum(r["usable_insight_present"] for r in rows),
        "zero_usable_insight": sum(r["zero_usable_insight"] for r in rows),
        "zero_insight_no_signal_observed": sum(
            r["zero_insight_no_signal_observed"] for r in rows
        ),
        "zero_insight_interviewer_failure": sum(
            r["zero_insight_interviewer_failure"] for r in rows
        ),
        "zero_insight_ai_loop": sum(r["zero_insight_ai_loop"] for r in rows),
        "clear_interviewing_defect": sum(
            r["clear_interviewing_defect"] for r in rows
        ),
        "forced_answer_repetition": sum(
            r["forced_answer_repetition"] for r in rows
        ),
        "employee_exact_repeat_any": sum(
            r["employee_exact_repeat_count"] > 0 for r in rows
        ),
        "employee_near_repeat_any": sum(
            r["employee_near_repeat_count"] > 0 for r in rows
        ),
        "signal_ever_present": sum(r["signal_ever_present"] for r in rows),
        "hard_defect_counts": dict(
            sorted(htypes.items(), key=lambda x: -x[1])
        ),
        "examples": {
            "no_insight_no_signal": [
                r["id"] for r in rows
                if r["conversation_outcome"] == "NO_INSIGHT_NO_SIGNAL"
            ][:12],

            "no_insight_ai_loop": [
                r["id"] for r in rows
                if r["conversation_outcome"] == "NO_INSIGHT_AI_LOOP"
            ][:12],

            "no_insight_missed_signal": [
                r["id"] for r in rows
                if r["conversation_outcome"] == "NO_INSIGHT_MISSED_SIGNAL"
            ][:12],

            "insightful_clean": [
                r["id"] for r in rows
                if r["conversation_outcome"] == "INSIGHTFUL_CLEAN"
            ][:12],

            "insightful_with_defects": [
                r["id"] for r in rows
                if r["conversation_outcome"] == "INSIGHTFUL_WITH_DEFECTS"
            ][:12],

            "forced_answer_repetition": [
                r["id"] for r in rows
                if r["forced_answer_repetition"]
            ][:12]
        },

        "note": (
            "Deterministic screening only. No-signal-observed does not prove "
            "the employee had nothing to add; insight classification should "
            "be human-validated."
        )
    }

    # Aggregate summary JSON
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=="__main__": main()
