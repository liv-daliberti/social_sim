"""Provenance recovery must never bypass template/checkpoint validation."""
from copy import deepcopy
import pytest
from exp1_prospective.context_reversal.audit_robustness_results import template_evidence


def fixture():
    config={'local_model':{'path':'checkpoint','metadata_sha256':{'tokenizer_config.json':'bound'}},'decode':{'chat_template_mode':'auto'}}
    manifests={'probability':{'status':'complete','config':deepcopy(config),'slurm_job_ids':['original','continuation']},
               'direction':{'status':'complete','config':deepcopy(config),'tokenizer_chat_template_sha256':'template'}}
    return manifests,[{'model_key':'model','original_job_id':'original','continuation_job_id':'continuation'}]


def test_direct_template_evidence():
    manifests,history=fixture()
    assert template_evidence('direction',manifests,'model',history,'template')['source']=='own_manifest'


def test_verified_completed_numeric_continuation():
    manifests,history=fixture()
    result=template_evidence('probability',manifests,'model',history,'template')
    assert result['source']=='paired_direction_manifest_after_numeric_completion_only_resume'
    assert result['per_response_chat_hashes_and_token_counts_still_required']


@pytest.mark.parametrize('fault',['wrong_template','different_checkpoint','different_mode','missing_job','unknown_history','missing_direction_hash','missing_direction_template'])
def test_inadequate_provenance_is_rejected(fault):
    manifests,history=fixture();task='probability';actual='template'
    if fault=='wrong_template':actual='changed'
    elif fault=='different_checkpoint':manifests['direction']['config']['local_model']['metadata_sha256']['tokenizer_config.json']='changed'
    elif fault=='different_mode':manifests['direction']['config']['decode']['chat_template_mode']='qwen-no-thinking'
    elif fault=='missing_job':manifests['probability']['slurm_job_ids']=['continuation']
    elif fault=='unknown_history':history=[]
    elif fault=='missing_direction_hash':del manifests['direction']['tokenizer_chat_template_sha256']
    elif fault=='missing_direction_template':
        del manifests['direction']['tokenizer_chat_template_sha256'];task='direction'
    with pytest.raises(ValueError):template_evidence(task,manifests,'model',history,actual)
