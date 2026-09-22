import json,sys,types
from pathlib import Path
import pytest
from exp1_prospective.context_reversal.fresh import reasoning as r,materials as m,freeze as f
from exp1_prospective.context_reversal import run_local as c


def test_interrupted_attempt_is_terminal_and_blocks_only_its_own_updates(tmp_path,monkeypatch):
    units=[m.unit(m.make_family(0,0),'base',ctx) for ctx in m.CONTEXTS]
    inp=tmp_path/'plan.jsonl';inp.write_text(''.join(json.dumps(x)+'\n' for x in units));fr=tmp_path/'freeze.json';fr.write_text('{}')
    args=r.build_parser().parse_args(['--model-path',str(tmp_path),'--model-key','test','--input',str(inp),'--output',str(tmp_path/'output.jsonl'),'--thinking','disabled','--batch-size','1','--max-tokens','128','--max-model-len','4096','--design-freeze',str(fr)])
    monkeypatch.setattr(c,'inspect_local_model',lambda p:{'path':str(p),'model_type':'qwen3'})
    monkeypatch.setattr(c,'describe_hardware',lambda:{'test':True})
    config=r.make_config(args,units);calls=[];crash=[True]
    class Tokenizer:
        chat_template='fake'
        def apply_chat_template(self,messages,**kwargs):return messages[0]['content']+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
        def encode(self,text,**kwargs):return list(range(100))
    class Engine:
        def __init__(self,**kwargs):pass
        def get_tokenizer(self):return Tokenizer()
        def generate(self,prompts,**kwargs):
            calls.append(len(prompts))
            if crash[0]:raise RuntimeError('simulated interruption after durable intent')
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='{"probability":0.5}',finish_reason='stop',stop_reason=None,token_ids=[1,2,3])]) for p in prompts]
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(LLM=Engine,SamplingParams=lambda **kw:kw))
    with pytest.raises(RuntimeError,match='simulated interruption'):r.execute(args,units,config)
    before=[json.loads(s) for s in args.output.read_text().splitlines()];assert len(before)==1 and before[0]['failure_kind']=='interrupted_unknown'
    unknown_key=c.record_key(before[0]);crash[0]=False;r.execute(args,units,config)
    rows=[json.loads(s) for s in args.output.read_text().splitlines()];assert len(rows)==12
    assert sum(x['status']=='ok' for x in rows)==8
    assert sum(x['status']=='blocked_baseline' for x in rows)==3
    assert next(x for x in rows if c.record_key(x)==unknown_key)['failure_kind']=='interrupted_unknown'
    assert sum(calls)==9  # one interrupted attempt, two other baselines, six updates
    prior_calls=len(calls);r.execute(args,units,config);assert len(calls)==prior_calls


def test_unfrozen_local_run_rejected(tmp_path):
    args=types.SimpleNamespace(design_freeze=None)
    with pytest.raises(ValueError,match='requires'):f.verify_local(args,{})
    p=tmp_path/'freeze.json';p.write_text('{"status":"draft"}');args.design_freeze=p
    with pytest.raises(ValueError,match='invalid design'):f.verify_local(args,{})


def test_artifact_change_rejected(tmp_path):
    p=tmp_path/'a';p.write_text('frozen');spec=f.artifact(p);p.write_text('changed')
    with pytest.raises(ValueError,match='changed'):f.check_artifact(spec)
