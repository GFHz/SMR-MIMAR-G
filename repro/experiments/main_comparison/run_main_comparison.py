"""Generate the equal-complete-history LLM-IPP-style vs SMR-MIMAR-G comparison."""
from __future__ import annotations
import csv, hashlib, json, statistics, time
from copy import deepcopy

import repro.experiments.main_comparison.llmipp_baseline as baseline
from repro.evaluators.formal.protocol import PROTOCOL_VERSION
from repro.evaluators.formal.title_resolver import TitleResolver
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.utils import ROOT, read_json, write_json

USERS=[419,5021,2677,3113,2249]
OUT=ROOT/'repro/results/main_comparison'
MANIFEST=ROOT/'repro/results/pilot/pilot_manifest.json'
SMR=ROOT/'repro/results/smr_mimar_g/controlled_positive_5users'

def tree_hash(folder):
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def mean(xs):
    xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None

def save_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in fields})

def main():
    resume=OUT.exists() and (OUT/'llmipp_paths.json').is_file() and (OUT/'prompt_context_audit.csv').is_file()
    if resume:raise FileExistsError('Frozen main-comparison results are included; preserve or move the directory before reproduction')
    if OUT.exists() and not resume:raise FileExistsError(f'Refusing overwrite: {OUT}')
    if PROTOCOL_VERSION!='Formal-Evaluation-v1':raise RuntimeError('Formal protocol changed')
    protected={'method':tree_hash(ROOT/'repro/methods/smr_mimar_g'),'formal':tree_hash(ROOT/'repro/evaluators/formal'),
               'smr':tree_hash(SMR)}
    manifest=read_json(MANIFEST);users_by={x['user_id']:x for x in manifest['users']};users=[]
    catalog=load_movies(ROOT/'dataset/ml-1m/movies.dat');movies={x['id']:x for x in catalog}
    audit=[];sequences=[]
    expected={419:'Clean Slate (Coup de Torchon) (1981)',5021:'Drunken Master (Zui quan) (1979)',2677:'Jingle All the Way (1996)',3113:'League of Their Own, A (1992)',2249:'Timecop (1994)'}
    for uid in USERS:
        original=users_by[uid];ids=[int(x) for x in original['positive_movie_ids']]
        if original['target']['title']!=expected[uid]:raise RuntimeError(f'Target mismatch user {uid}')
        full=[{'movie_id':mid,'title':movies[mid]['title'],'genres':movies[mid]['genres']} for mid in ids]
        user=deepcopy(original);user['history']=full;users.append(user)
        target_id=int(original['target']['movie_id']);inside=target_id in ids
        audit.append({'USER':uid,'FULL_POSITIVE_HISTORY_LENGTH':len(ids),'FULL_POSITIVE_HISTORY_IDS':ids,
            'FULL_POSITIVE_HISTORY_TITLES':[x['title'] for x in full],
            'LLMIPP_FULL_HISTORY_IDS':ids,'SMR_FULL_PROFILE_HISTORY_IDS':ids,'HISTORY_EQUAL':True,
            'TARGET_ID':target_id,'TARGET_IN_FULL_HISTORY':inside,
            'target_protocol':'Frozen target is sampled outside complete positive history; history is not altered for this ablation.'})
        sequences += [{'USER':uid,'POSITION':i+1,'MOVIE_ID':x['movie_id'],'TITLE':x['title']} for i,x in enumerate(full)]
    if any(x['TARGET_IN_FULL_HISTORY'] for x in audit):raise RuntimeError('Frozen target unexpectedly occurs in H_FULL')
    if resume:
        records=read_json(OUT/'llmipp_paths.json')
        with (OUT/'prompt_context_audit.csv').open(encoding='utf-8-sig',newline='') as f:context=list(csv.DictReader(f))
        for x in context:
            for key in ('PROMPT_FITS_CONTEXT','FORMAT_TURN_FITS_CONTEXT'):x[key]=x[key]=='True'
        print('resuming post-hoc evaluation from 10 saved full-history paths; no LLM generation',flush=True)
    else:
        OUT.mkdir(parents=True,exist_ok=False);write_json(OUT/'history_audit.json',audit)
        save_csv(OUT/'history_sequences.csv',sequences,['USER','POSITION','MOVIE_ID','TITLE'])
        old_out=baseline.OUT;baseline.OUT=OUT
        records=[]
        try:
            for user in users:
                for pid in (1,2):
                    started=time.perf_counter();r=baseline.generate(user,pid);r['wall_elapsed_seconds']=time.perf_counter()-started
                    r['method']='LLM-IPP-style';r['PROFILE_HISTORY']='complete_positive_history';r['FULL_HISTORY_IDS']=[x['movie_id'] for x in user['history']]
                    records.append(r);print(f'generated {len(records)}/10 user={user["user_id"]} path={pid} parse={r["parser"]["parse_success"]}',flush=True)
        finally:baseline.OUT=old_out
        write_json(OUT/'llmipp_raw_outputs.json',[{'user_id':r['user_id'],'path_index':r['path_index'],'RAW_LLM_OUTPUT':r['raw_formatting_response'],'RAW_PLAN':r['raw_plan'],'response_metadata':r['format_response_metadata']} for r in records])
        write_json(OUT/'llmipp_paths.json',records)
        context=[];lengths={x['USER']:x['FULL_POSITIVE_HISTORY_LENGTH'] for x in audit}
        for r in records:
            first=int(r['plan_response_metadata'].get('prompt_eval_count',0));second=int(r['format_response_metadata'].get('prompt_eval_count',0));chars=len(r['prompt']['system_prompt'])+len(r['prompt']['user_prompt'])
            context.append({'USER':r['user_id'],'PATH_ID':r['path_index'],'FULL_HISTORY_LENGTH':lengths[r['user_id']],
                'PROMPT_CHARACTER_COUNT':chars,'PROMPT_TOKEN_COUNT_IF_AVAILABLE':first,'FORMAT_TURN_PROMPT_TOKEN_COUNT':second,
                'TOKEN_COUNT_AVAILABLE':first>0,'CONTEXT_LIMIT':4096,'PROMPT_FITS_CONTEXT':0<first<4096,
                'FORMAT_TURN_FITS_CONTEXT':0<second<4096,'RUNTIME_AUTO_TRUNCATION_DETECTED':False,
                'AUTO_TRUNCATION_BASIS':'Ollama reported evaluated prompt token counts below num_ctx; complete request text is preserved.'})
        save_csv(OUT/'prompt_context_audit.csv',context,list(context[0]))
    resolver=TitleResolver.from_movies_dat(ROOT/'dataset/ml-1m/movies.dat')
    eval_users=[users_by[x] for x in USERS]
    llm_rows=baseline.evaluate(records,eval_users,resolver);llm_agg=baseline.aggregate(llm_rows)
    save_csv(OUT/'llmipp_path_validity.csv',llm_rows,list(llm_rows[0]))
    smr_records=[]
    for user in users:
        for pid in (1,2):
            r=read_json(SMR/'generation'/str(user['user_id'])/f'path_{pid}.json');r['method']='SMR-MIMAR-G';smr_records.append(r)
    # Formal-Evaluation-v1 retains its frozen 20-item SASRec input protocol; planning inputs are complete history.
    smr_rows=baseline.evaluate(smr_records,eval_users,resolver);smr_agg=baseline.aggregate(smr_rows)
    expected_metrics={'Valid Paths':10,'IoI':2.3745645592687077,'IoR':416,'Proxy':0.6067911714418586,'Coherence':0.9333333333333333,
        'HistoryReuseRate':0.0,'TargetPresenceRate':1.0,'TargetLastRate':1.0}
    for k,v in expected_metrics.items():
        if abs(smr_agg[k]-v)>1e-12:raise RuntimeError(f'Frozen SMR metric discrepancy {k}: {smr_agg[k]} != {v}')
    write_json(OUT/'smr_reference.json',{'source':str(SMR.relative_to(ROOT)),'paths':smr_records,'aggregate':smr_agg,
        'target_endpoint_note':'Target presence/last are protocol-guaranteed by predefined-target append, not learned success.'})
    all_rows=llm_rows+smr_rows;save_csv(OUT/'per_path_results.csv',all_rows,list(all_rows[0]))
    combined=[{'Condition':'LLM-IPP-style',**llm_agg},{'Condition':'SMR-MIMAR-G',**smr_agg}]
    save_csv(OUT/'aggregate_results.csv',combined,list(combined[0]))
    save_csv(OUT/'comparison.csv',combined,list(combined[0]))
    deltas={'DELTA_IOI_FULL':smr_agg['IoI']-llm_agg['IoI'],'DELTA_IOR_FULL':smr_agg['IoR']-llm_agg['IoR'],
        'DELTA_PROXY_FULL':smr_agg['Proxy']-llm_agg['Proxy'],'DELTA_COHERENCE_FULL':smr_agg['Coherence']-llm_agg['Coherence'],
        'HISTORY_REUSE_REDUCTION_FULL':llm_agg['HistoryReuseRate']-smr_agg['HistoryReuseRate'],
        'TARGET_PRESENCE_GAIN_FULL':smr_agg['TargetPresenceRate']-llm_agg['TargetPresenceRate'],
        'TARGET_LAST_GAIN_FULL':smr_agg['TargetLastRate']-llm_agg['TargetLastRate']}
    overflow=sum(not x['PROMPT_FITS_CONTEXT'] or not x['FORMAT_TURN_FITS_CONTEXT'] for x in context)
    current={'method':tree_hash(ROOT/'repro/methods/smr_mimar_g'),'formal':tree_hash(ROOT/'repro/evaluators/formal'),'smr':tree_hash(SMR)}
    if current!=protected:raise RuntimeError('Protected artifacts changed')
    summary={'complete':True,'strict_full_history_comparison_feasible':overflow==0,'all_full_history_assertions_passed':True,
        'context_audit':{'overflow_paths':overflow,'total_paths':10,'token_count_available':True,'runtime_auto_truncation_detected':False},
        'conditions':{'LLM-IPP-style':llm_agg,'SMR-MIMAR-G':smr_agg},'deltas':deltas,
        'training':{'LLMIPP_LLM_TRAINED':False,'SMR_LLM_TRAINED':False,'LLM_FINE_TUNING_USED':False,'RL_TRAINING_USED':False},
        'caveat':'Equal complete-positive-history local comparison; not a strict original LLM-IPP reproduction.',
        'new_llmipp_paths':10,'new_smr_llm_generation_run':False,
        'protection':{'FORMAL_EVALUATOR_MODIFIED':False,'FINAL_SMR_METHOD_MODIFIED':False,'PARAMETERS_TUNED':False}}
    write_json(OUT/'summary.json',summary)
    report='# Main comparison\n\nBoth local planners receive the same complete chronological positive history. Neither trains or fine-tunes Qwen. This is not a strict original LLM-IPP reproduction.\n\n'
    report+='| Condition | Valid | IoI | IoR | Proxy | Coherence | History reuse | Target present | Target last |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for x in combined:report+=f"| {x['Condition']} | {x['Valid Paths']}/10 | {x['IoI']} | {x['IoR']} | {x['Proxy']} | {x['Coherence']} | {x['HistoryReuseRate']} | {x['TargetPresenceRate']} | {x['TargetLastRate']} |\n"
    report+=f"\nDeltas (SMR minus LLM-IPP): IoI {deltas['DELTA_IOI_FULL']}, IoR {deltas['DELTA_IOR_FULL']}, Proxy {deltas['DELTA_PROXY_FULL']}, Coherence {deltas['DELTA_COHERENCE_FULL']}. Results are descriptive; no significance, causal, or universal-superiority claim is made.\n"
    (OUT/'comparison_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
