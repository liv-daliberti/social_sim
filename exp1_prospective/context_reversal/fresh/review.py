"""Blinded separate-checkpoint screening. Never loads labels or target responses."""
import argparse,json,os,re
from pathlib import Path
from .. import run_local as common


def parse(raw,thinking,finish):
    if finish!='stop':raise ValueError('nonterminal or truncated completion')
    final=raw
    if thinking:
        before,sep,final=raw.partition('</think>')
        if not sep:raise ValueError('unclosed thinking')
    final=final.strip()
    fence=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',final)
    if fence:final=fence.group(1)
    value=json.loads(final)
    if set(value)!={'direction','endpoints_possible','concern'}:raise ValueError('review fields')
    if value['direction'] not in ('increase','decrease','unchanged','unclear') or type(value['endpoints_possible']) is not bool or not isinstance(value['concern'],str):raise ValueError('review values')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--packets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);p.add_argument('--thinking',action='store_true');p.add_argument('--tp',type=int,default=4);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    packets=[json.loads(s) for s in a.packets.read_text().splitlines()]
    assert len({r['review_id'] for r in packets})==len(packets)
    assert all(set(r)=={'review_id','prompt'} for r in packets)
    cfg={'packets_path':str(a.packets.resolve()),'packets_sha256':common.file_sha256(a.packets),'model':common.inspect_local_model(a.model_path),'thinking':a.thinking,'max_tokens':4096 if a.thinking else 1024,'temperature':0,'seed':20260922,'batch_size':a.batch_size,'tp':a.tp,'runner_sha256':common.file_sha256(Path(__file__)),'helper_sha256':common.file_sha256(Path(common.__file__))}
    sig=common.text_sha256(common.canonical_json(cfg));mp=Path(str(a.output)+'.manifest.json')
    rows={}
    if mp.exists():assert json.loads(mp.read_text())['signature']==sig
    elif a.output.exists():raise ValueError('Unbound review output')
    if a.output.exists():
        for s in a.output.read_text().splitlines():
            r=json.loads(s);assert r['review_id'] not in rows and r['signature']==sig;rows[r['review_id']]=r
    if a.dry_run:print(json.dumps({'status':'valid','packets':len(packets),'existing':len(rows),'config':cfg}));return
    with common.output_lock(Path(str(a.output)+'.lock')):
        manifest={'signature':sig,'config':cfg,'expected':len(packets),'status':'running','job_id':os.getenv('SLURM_JOB_ID'),'started_at':common.utc_now()}
        def save():
            manifest.update(received=len(rows),updated_at=common.utc_now(),valid=sum(r['status']=='ok' for r in rows.values()))
            common.atomic_write(a.output,''.join(common.canonical_json(r)+'\n' for r in rows.values()))
            common.atomic_write(mp,json.dumps(manifest,indent=2)+'\n')
        save()
        from vllm import LLM,SamplingParams
        engine=LLM(model=str(a.model_path),tokenizer=str(a.model_path),trust_remote_code=False,tensor_parallel_size=a.tp,max_model_len=8192,gpu_memory_utilization=.9,dtype='bfloat16',seed=20260922,enable_prefix_caching=True,enforce_eager=True,max_num_seqs=a.batch_size,disable_custom_all_reduce=True)
        tok=engine.get_tokenizer();manifest['chat_template_sha256']=common.text_sha256(common.canonical_json(tok.chat_template));manifest['hardware']=common.describe_hardware()
        pending=[r for r in packets if r['review_id'] not in rows]
        for start in range(0,len(pending),a.batch_size):
            batch=pending[start:start+a.batch_size]
            chats=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=a.thinking) for r in batch]
            params=[SamplingParams(temperature=0,top_p=1,max_tokens=cfg['max_tokens'],seed=common.request_seed(20260922,r['review_id'],'review','review'),skip_special_tokens=False) for r in batch]
            outputs=engine.generate(chats,params,use_tqdm=False)
            for r,chat,out in zip(batch,chats,outputs):
                o=out.outputs[0];raw=o.text
                # Remove only known assistant terminators, not arbitrary content.
                for ending in ('<|im_end|>','<|eot_id|>','<|end_of_text|>','</s>'):
                    if raw.rstrip().endswith(ending):raw=raw.rstrip()[:-len(ending)]
                row={'review_id':r['review_id'],'signature':sig,'prompt_sha256':common.text_sha256(r['prompt']),'chat_sha256':common.text_sha256(chat),'raw':raw,'finish_reason':o.finish_reason,'input_tokens':len(out.prompt_token_ids),'output_tokens':len(o.token_ids),'status':'parse_error','judgment':None}
                try:row.update(judgment=parse(raw,a.thinking,o.finish_reason),status='ok')
                except (ValueError,TypeError) as e:row['error']=str(e)
                rows[r['review_id']]=row
            save();print(f'review checkpoint {len(rows)}/{len(packets)}',flush=True)
        manifest['status']='complete';save()

if __name__=='__main__':main()
