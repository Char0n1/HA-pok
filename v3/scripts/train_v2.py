"""BO1 online PPO with learned optional/multiple selections and fresh rollouts."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ptcg_rl import v2
from ptcg_rl.policy import CandidateActorCritic
from ptcg_rl.ppo import PPOBatch, ppo_update


def main():
    global v2
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--feature-version', choices=['v2','v3'], default='v3')
    parser.add_argument('--competition-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sdk = args.competition_root.resolve()/'sample_submission'/'sample_submission'/'sample_submission'
    sys.path.insert(0, str(sdk))
    from cg.game import battle_start, battle_select, battle_finish
    catalogs = {}
    if args.feature_version == 'v3':
        from ptcg_rl import v3 as v2
        from cg.sim import lib
        catalogs = dict(cards={x['cardId']:x for x in json.loads(lib.AllCard())},
                        attacks={x['attackId']:x for x in json.loads(lib.AllAttack())})
        v2.configure(**catalogs)
    args.output = args.output or ROOT/'artifacts'/f'{args.feature_version}_smoke.pt'
    deck = [int(x) for x in (sdk/'deck.csv').read_text().split()]
    model = CandidateActorCritic(state_dim=v2.STATE_DIM, option_dim=v2.INPUT_OPTION_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = []; totals = {'completed':0, 'truncated':0, 'wins':[0,0], 'draws':0, 'decisions':0, 'updates':0}
    started = time.monotonic()

    @torch.inference_mode()
    def score(state, options):
        logits, value = model(torch.from_numpy(state)[None], torch.from_numpy(options)[None])
        return logits[0].numpy(), value.item()

    def update():
        if not buffer: return
        # Same weights generated all old probabilities; verify before updating.
        for row in buffer[:10]:
            logits, _ = score(row[0], row[1]); p = torch.distributions.Categorical(logits=torch.tensor(logits))
            assert abs(p.log_prob(torch.tensor(row[2])).item()-row[3]) < 1e-4
        for epoch in range(2):
            for offset in range(0, len(buffer), 128):
                rows = buffer[offset:offset+128]; n = max(len(r[1]) for r in rows)
                options = np.zeros((len(rows), n, v2.INPUT_OPTION_DIM), np.float32)
                masks = np.zeros((len(rows), n), bool)
                for i, r in enumerate(rows): options[i,:len(r[1])] = r[1]; masks[i,:len(r[1])] = True
                returns = torch.tensor([r[5] for r in rows], dtype=torch.float32)
                batch = PPOBatch(torch.tensor(np.stack([r[0] for r in rows])), torch.tensor(options), torch.tensor(masks),
                    torch.tensor([r[2] for r in rows]), torch.tensor([r[3] for r in rows]), returns,
                    returns-torch.tensor([r[4] for r in rows]))
                metrics = ppo_update(model, optimizer, batch)
                assert all(np.isfinite(x) for x in metrics.values())
                totals['updates'] += 1
        buffer.clear()

    for game in range(args.games):
        obs, _ = battle_start(deck, deck)
        trajectory = []
        try:
            for step in range(2000):
                winner = obs['current']['result']
                if winner != -1: break
                actor = obs['current']['yourIndex']
                action, traces = v2.choose(obs, score, True, rng)
                trajectory.extend((t, actor) for t in traces)
                obs = battle_select(action)
            winner = obs['current']['result']
            if winner == -1:
                totals['truncated'] += 1
                # Discard incomplete games: do not label timeouts as losses.
            else:
                totals['completed'] += 1
                if winner in (0,1): totals['wins'][winner] += 1
                else: totals['draws'] += 1
                for trace, actor in trajectory:
                    reward = (1. if winner == actor else -1.) if winner in (0,1) else 0.
                    buffer.append((*trace, reward))
                totals['decisions'] += len(trajectory)
        finally:
            battle_finish()
        if len(buffer) >= 1024: update()
        if (game+1)%10 == 0: print(json.dumps(dict(game=game+1, **totals)), flush=True)
    update()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(schema=v2.SCHEMA, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        stats=totals, catalogs=catalogs, seconds=time.monotonic()-started, deck=deck, seed=args.seed,
        objective='BO1 terminal reward, Monte Carlo returns, no BO3 memory'), args.output)
    print('Saved', args.output, totals, flush=True)

if __name__ == '__main__': main()
