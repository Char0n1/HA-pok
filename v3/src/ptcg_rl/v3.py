"""V3 observable discard identities and engine-backed attack features.

Damage fields are pre-modifier components, NOT a complete damage simulator.
Catalogs are frozen in checkpoints and embedded in exported submissions.
"""
import numpy as np
from . import v2

SCHEMA = 'ptcg-v3-discard-attacks-1'
STATE_DIM = v2.STATE_DIM + 60*16 + 12
OPTION_DIM = v2.OPTION_DIM + 20
INPUT_OPTION_DIM = OPTION_DIM*2
CARDS = {}
ATTACKS = {}


def configure(cards, attacks):
    global CARDS, ATTACKS
    CARDS = {int(k): v for k, v in cards.items()}
    ATTACKS = {int(k): v for k, v in attacks.items()}
    if not CARDS or not ATTACKS:
        raise ValueError('V3 requires frozen engine catalogs')


def discard_counts(player):
    counts = np.zeros(12, dtype=np.float32)
    for card in player.get('discard') or []:
        meta = CARDS.get((card or {}).get('id'), {})
        energy = meta.get('energyType')
        if meta.get('cardType') == 5 and isinstance(energy, int) and 0 <= energy < 12:
            counts[energy] += 1
    return counts


def encode_state(obs):
    if not CARDS:
        raise RuntimeError('Configure V3 catalogs before encoding')
    c = obs['current']; own = c['players'][c['yourIndex']]
    ids = sorted(x['id'] for x in own.get('discard', []) if x)
    encoded = [bit for i in range(60) for bit in v2.bits(ids[i] if i < len(ids) else 0)]
    return np.concatenate([v2.encode_state(obs), encoded, discard_counts(own)/60]).astype(np.float32)


def option_features(obs, op):
    # New features precede the four progress slots used by sequential selection.
    base = v2.option_features(obs, op)
    extra = np.zeros(20, dtype=np.float32)
    if op.get('type') == 13:
        attack = ATTACKS.get(op.get('attackId'))
        if attack:
            extra[0] = 1  # metadata present, distinct from unknown attack
            extra[1] = attack.get('damage', 0)/600
            for energy in attack.get('energies', []):
                if 0 <= energy < 12: extra[2+energy] += 1/10
            c = obs['current']; own = c['players'][c['yourIndex']]
            # Explicit engine-ID + name guards; no speculative generic text parsing.
            if op.get('attackId') == 1042 and attack.get('name') == 'Riptide':
                extra[14] = 1  # observable variable damage formula known
                extra[15] = 20*discard_counts(own)[3]/600
            if op.get('attackId') == 1046 and attack.get('name') == 'Hammer-lanche':
                extra[16] = 1  # discards top six; random damage, not known final damage
                extra[17] = min(6, own.get('deckCount', 0))/60
                extra[18] = float(own.get('deckCount', 0) <= 6)
            extra[19] = float(bool(attack.get('text')))  # effects may modify printed damage
    return np.concatenate([base[:-4], extra, base[-4:]]).astype(np.float32)


def selection_frame(obs, selected):
    s = obs['select']; count = len(selected)
    mapping = [i for i in range(len(s['option'])) if i not in selected]
    rows = [option_features(obs, s['option'][i]) for i in mapping]
    if count >= s['minCount']:
        mapping.append(-1); rows.append(option_features(obs, {'type':17}))
    options = np.stack(rows)
    prefix = np.mean([option_features(obs, s['option'][i]) for i in selected],axis=0) if selected else np.zeros(OPTION_DIM)
    state = encode_state(obs)
    state[6:8] = [(s['minCount']-count)/60, (s['maxCount']-count)/60]
    options[:,-4:] = [count/60, max(s['minCount']-count,0)/60,(s['maxCount']-count)/60,len(mapping)/100]
    return state, np.concatenate([options,np.repeat(prefix[None],len(rows),axis=0)],axis=1).astype(np.float32), mapping


def choose(obs, score, stochastic=False, rng=None):
    return v2.choose(obs, score, stochastic, rng, frame_fn=selection_frame)
