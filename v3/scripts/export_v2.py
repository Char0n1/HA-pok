"""Export v2 using the exact same feature and selection code as training."""
import argparse
import sys
import tarfile
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ptcg_rl import v2
from ptcg_rl.policy import CandidateActorCritic

RUNTIME = '''
import os
_weights = None
def resource_path(name):
    for base in ('/kaggle_simulations/agent', os.getcwd()):
        path = os.path.join(base,name)
        if os.path.isfile(path): return path
    raise FileNotFoundError(name)
def score(state, options):
    global _weights
    if _weights is None: _weights = np.load(resource_path('model.npz'))
    def linear(x,key): return x @ _weights[key+'.weight'].T + _weights[key+'.bias']
    h=np.tanh(linear(np.tanh(linear(state,'state_encoder.0')),'state_encoder.2'))
    a=np.tanh(linear(options,'option_encoder.0'))
    logits=linear(np.tanh(linear(np.concatenate([np.repeat(h[None],len(a),axis=0),a],axis=1),'logit_head.0')),'logit_head.2').ravel()
    value=linear(np.tanh(linear(h,'value_head.0')),'value_head.2').item()
    return logits,value
def read_deck():
    with open(resource_path('deck.csv')) as f: return [int(x) for x in f.read().split()]
def agent(obs_dict):
    if obs_dict.get('select') is None: return read_deck()
    return choose(obs_dict,score)[0]
'''

def main():
    global v2
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    version = 'v2'
    if checkpoint.get('schema') == 'ptcg-v3-discard-attacks-1':
        from ptcg_rl import v3 as v2
        v2.configure(**checkpoint['catalogs'])
        version = 'v3'
    if checkpoint.get('schema') != v2.SCHEMA: raise ValueError('Incompatible checkpoint schema')
    model=CandidateActorCritic(state_dim=v2.STATE_DIM,option_dim=v2.INPUT_OPTION_DIM)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    folder=ROOT/f'submission_{version}'; folder.mkdir(exist_ok=True)
    source=(ROOT/'src'/'ptcg_rl'/'v2.py').read_text(encoding='utf-8')+RUNTIME
    if version == 'v3':
        import json
        base=(ROOT/'src'/'ptcg_rl'/'v2.py').read_text(encoding='utf-8')
        extension=(ROOT/'src'/'ptcg_rl'/'v3.py').read_text(encoding='utf-8').replace('from . import v2','v2 = _base')
        source = ('import types, json\n_base = types.ModuleType("v2_base")\nexec('+repr(base)+', _base.__dict__)\n'
                  + extension + '\nconfigure(**json.loads('+repr(json.dumps(checkpoint['catalogs']))+'))\n'+RUNTIME)
    (folder/'main.py').write_text(source,encoding='utf-8')
    (folder/'deck.csv').write_text('\n'.join(map(str,checkpoint['deck']))+'\n',encoding='utf-8')
    np.savez_compressed(folder/'model.npz',**{k:v.numpy() for k,v in model.state_dict().items()})
    namespace={}; exec(compile(source,'main.py','exec'),namespace)
    namespace['_weights']=np.load(folder/'model.npz')
    checked=0
    # Numerical parity is self-contained; game semantics are tested separately.
    rng=np.random.default_rng(42)
    for count in range(1,41):
        state=rng.normal(size=v2.STATE_DIM).astype(np.float32)
        options=rng.normal(size=(count,v2.INPUT_OPTION_DIM)).astype(np.float32)
        with torch.inference_mode():
            logits,value=model(torch.tensor(state)[None],torch.tensor(options)[None])
        actual,actual_value=namespace['score'](state,options)
        np.testing.assert_allclose(actual,logits[0].numpy(),atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(actual_value,value.item(),atol=2e-5,rtol=2e-5)
        checked+=1
    with tarfile.open(folder/'submission.tar.gz','w:gz') as bundle:
        for name in ('main.py','deck.csv','model.npz'): bundle.add(folder/name,arcname=name)
    print('Exported',folder,'; NumPy/PyTorch parity decisions:',checked)

if __name__=='__main__': main()
