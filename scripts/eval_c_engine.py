import json, random, re, subprocess, sys
from collections import Counter
from pathlib import Path
ROOT=Path('/Users/M2/codes/esp32/LILM_V1')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'schema'))
from dsl import Action, actions_match
RUNNER='/tmp/n2esp32/build/host_runner'
SCHEMA=str(ROOT/'schema/tool_schema.json')
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
CACT=sys.argv[2] if len(sys.argv)>2 else str(ROOT/'robot.cact')
REPAIR='--repair' in sys.argv

def repair_calls(calls):
    out=[]
    for c in calls:
        args=dict(c.get('arguments') or {}); intent=args.pop('intent','?')
        if intent=='WAKEUP' and 'recipient' not in args and 'duration_amount' in args:
            intent='WAIT'
        out.append({'intent':intent,'slots':{k:str(v) for k,v in args.items()}})
    return out
records=[json.loads(l) for l in open(ROOT/'data/finetune/train_v2.jsonl',encoding='utf-8') if l.strip()]
random.seed(42)
sample=random.sample(records,min(N,len(records)))
def gold_actions(rec):
    out=[]
    for a in rec.get('answers',[]):
        args=dict(a.get('arguments') or {}); intent=args.pop('intent','?')
        try: out.append(Action(intent=intent,slots={k:str(v) for k,v in args.items()}))
        except ValueError: pass
    return out
ok=0; per=Counter(); per_ok=Counter(); fails=[]
for i,rec in enumerate(sample):
    gold=gold_actions(rec)
    r=subprocess.run([RUNNER,CACT,SCHEMA,rec['query']],capture_output=True,text=True,timeout=300)
    txt=r.stdout.replace('OUT: ','').strip().replace('▁',' ')
    pred=None; err=None
    m=re.search(r'<tool_call>(\[.*?\])</tool_call>',txt,re.S)
    if not m: err='no tool_call'
    else:
        try:
            calls=json.loads(m.group(1))
            if REPAIR: calls=repair_calls(calls)
            pred=[]
            for c in calls:
                intent=c.get('intent','?'); slots=dict(c.get('slots') or {})
                pred.append(Action.from_dict({'intent':intent,'slots':slots}))
        except Exception as e: err=repr(e); pred=None
    mm=actions_match(pred,gold)
    ok+=mm
    for g in gold:
        per[g.intent]+=1
        if mm: per_ok[g.intent]+=1
    if not mm and len(fails)<10:
        fails.append((rec['query'],[g.to_dict() for g in gold], txt[:160]))
    if (i+1)%25==0: print(f'  ... {i+1}/{len(sample)} acc {ok}/{i+1}',flush=True)
print('\n'+'='*60)
print(f'C99-ENGINE EVAL: {ok}/{len(sample)} = {100*ok/len(sample):.1f}% exact match\n')
for k in sorted(per): print(f'  {k:<12} {per_ok[k]}/{per[k]}')
print('\nSample fails:')
for q,g,t in fails: print(f'\nQ {q!r}\n  gold {g}\n  got  {t!r}')
