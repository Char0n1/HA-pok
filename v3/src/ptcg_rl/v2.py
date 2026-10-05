"""V2: observable entity features and sequential selection with learned STOP.

Pure NumPy feature code is shared by training and exported inference.
IDs use lossless 16-bit representations instead of modulo buckets. This is a
small baseline, not a learned card-text encoder or a recurrent belief model.
"""
import numpy as np

SCHEMA = 'ptcg-v2-sequential-1'
CARD_DIM = 32
STATE_DIM = 24 + 64 + 12 * CARD_DIM + 60 * 16 + 60 * 16
OPTION_DIM = 18 + 64 + 3 * CARD_DIM + 16 + 14


def bits(value):
    value = int(value or 0)
    if not 0 <= value < 65536:
        raise ValueError('ID exceeds feature schema')
    return [(value >> i) & 1 for i in range(16)]


def onehot(value, size):
    out = [0.] * size
    out[min(max(int(value or 0), 0), size - 1)] = 1.
    return out


def card(c):
    c = c or {}
    energies = c.get('energies') or []
    counts = [energies.count(i) / 8 for i in range(12)]
    return bits(c.get('id', c.get('cardId', 0))) + [c.get('hp', 0) / 400,
        c.get('maxHp', 0) / 400, float(c.get('appearThisTurn', False)),
        len(c.get('tools') or []) / 4] + counts


def resolve(obs, area, index, player):
    current = obs.get('current') or {}
    players = current.get('players') or [{}, {}]
    if player not in (0, 1):
        return None
    p = players[player]
    if area == 1:
        # The engine reveals this only for appropriate search effects.
        values = (obs.get('select') or {}).get('deck') or []
    elif area == 7:
        values = current.get('stadium') or []
        if isinstance(values, dict):
            values = [values]
    elif area == 12:
        values = current.get('looking') or []
    else:
        key = {2: 'hand', 3: 'discard', 4: 'active', 5: 'bench', 6: 'prize'}.get(area)
        if key == 'hand' and player != current.get('yourIndex', 0):
            return None
        values = p.get(key) or []
    return values[index] if isinstance(index, int) and 0 <= index < len(values) else None


def encode_state(obs):
    c = obs.get('current') or {}
    s = obs.get('select') or {}
    me = c.get('yourIndex', 0)
    players = c.get('players') or [{}, {}]
    own, opp = players[me], players[1-me]
    vals = [c.get('turn', 0)/30, c.get('turnActionCount', 0)/30,
        float(c.get('firstPlayer') == me), c.get('round', 1)/3,
        own.get('win', 0)/2, opp.get('win', 0)/2,
        s.get('minCount', 0)/60, s.get('maxCount', 0)/60]
    vals += [float(bool(c.get(k))) for k in ('supporterPlayed', 'stadiumPlayed', 'energyAttached', 'retreated')]
    for p in (own, opp):
        vals += [p.get('handCount', 0)/60, p.get('deckCount', 0)/60,
                 len(p.get('prize') or []) / 6, len(p.get('bench') or []) / 8,
                 len(p.get('discard') or []) / 60,
                 sum(bool(p.get(k)) for k in ('poisoned','burned','asleep','paralyzed','confused'))/5]
    vals += onehot(s.get('context'), 64)
    for p in (own, opp):
        active = p.get('active') or []
        bench = p.get('bench') or []
        # Six slots cover the default board. Expanded benches are not fully
        # encoded here; option target features still resolve every candidate.
        vals += card(active[0] if active else None)
        for i in range(5):
            vals += card(bench[i] if i < len(bench) else None)
    hand = sorted(x['id'] for x in (own.get('hand') or []) if x)
    known_discard = sorted(x['id'] for x in (opp.get('discard') or []) if x)
    for ids in (hand, known_discard):
        for i in range(60):
            vals += bits(ids[i] if i < len(ids) else 0)
    return np.asarray(vals, dtype=np.float32)


def option_features(obs, op):
    s = obs['select']; me = obs['current']['yourIndex']
    player = op.get('playerIndex', me)
    kind = op.get('type', 17)
    area = op.get('area', 2 if kind == 7 else 0)
    source = resolve(obs, area, op.get('index', -1), player)
    target = resolve(obs, op.get('inPlayArea', 0), op.get('inPlayIndex', -1), me)
    if kind in (4, 5):
        attached = (source or {}).get('tools' if kind == 4 else 'energyCards') or []
        index = op.get('toolIndex' if kind == 4 else 'energyIndex', -1)
        target, source = source, attached[index] if 0 <= index < len(attached) else None
    if kind == 13:
        source = resolve(obs, 4, 0, me)
        target = resolve(obs, 4, 0, 1-me)
    if source is None and op.get('cardId'):
        source = {'id': op['cardId']}
    vals = onehot(kind, 18) + onehot(s.get('context'), 64)
    vals += card(source) + card(target) + card(s.get('effect') or s.get('contextCard'))
    vals += bits(op.get('attackId'))
    vals += [area/12, op.get('inPlayArea', 0)/12, float(player == me),
             op.get('number', 0)/60, op.get('count', 0)/20,
             op.get('specialConditionType', 0)/5,
             s.get('type', 0)/10, s.get('remainDamageCounter', 0)/30,
             s.get('remainEnergyCost', 0)/10, float(kind == 17),
             0., 0., 0., 0.]
    return np.asarray(vals, dtype=np.float32)


def selection_frame(obs, selected):
    """A micro-action is a remaining candidate or STOP once minCount is met."""
    s = obs['select']; count = len(selected)
    remaining = [i for i in range(len(s['option'])) if i not in selected]
    rows = [option_features(obs, s['option'][i]) for i in remaining]
    if count >= s['minCount']:
        remaining.append(-1)
        rows.append(option_features(obs, {'type': 17}))
    options = np.stack(rows)
    # Condition each next choice on the selected prefix, without replacement.
    prefix = np.mean([option_features(obs, s['option'][i]) for i in selected], axis=0) if selected else np.zeros(OPTION_DIM)
    state = encode_state(obs)
    state[6:8] = [(s['minCount']-count)/60, (s['maxCount']-count)/60]
    options[:, -4:] = [count/60, max(s['minCount']-count, 0)/60, (s['maxCount']-count)/60, len(remaining)/100]
    # Preserve current candidate identity and expose the chosen cards separately.
    return state, np.concatenate([options, np.repeat(prefix[None], len(rows), axis=0)], axis=1).astype(np.float32), remaining


INPUT_OPTION_DIM = OPTION_DIM * 2


def choose(obs, score, stochastic=False, rng=None, frame_fn=None):
    selected, traces = [], []
    s = obs['select']
    if not 0 <= s['minCount'] <= s['maxCount'] <= len(s['option']):
        raise ValueError('Invalid selection bounds')
    while len(selected) < s['maxCount']:
        state, options, mapping = (frame_fn or selection_frame)(obs, selected)
        logits, value = score(state, options)
        logits = np.asarray(logits, dtype=np.float64)
        probs = np.exp(logits - np.max(logits)); probs /= probs.sum()
        index = int(rng.choice(len(mapping), p=probs)) if stochastic else int(np.argmax(logits))
        traces.append((state, options, index, float(np.log(probs[index])), float(value)))
        if mapping[index] == -1:
            break
        selected.append(mapping[index])
    return selected, traces
