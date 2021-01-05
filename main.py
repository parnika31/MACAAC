import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.atten_sac_cons_2 import AttentionSAC
from random import seed


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    seed(config.seed)
    if config.use_gpu:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # initialize lagrange parameter
    lb_t = torch.from_numpy(np.random.rand(1)).float()

    all_rewards = []
    all_penalties = []
    b_t = config.b_t  # slower timescale value
    alpha = config.alpha  # threshold value

    # penalty flag for agents collision (refer to simple_spread.py)
    extra_cost = 1000

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir))

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        episode_rewards = []
        episode_penalties = []

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            ls_rollout_rewards = []
            ls_rollout_penalties = []
            penalties = []
            for reward in rewards:
                penalty = 0
                if (reward[0] - extra_cost) > 0:
                    for idx in range(len(reward)):
                        reward[idx] -= extra_cost
                    penalty = 1
                    ls_rollout_rewards.append(reward[0])
                    ls_rollout_penalties.append(penalty)
                    penalties.append(np.asarray([penalty] * config.n_agents))
                    for idx in range(len(reward)):
                        reward[idx] += lb_t * penalty
                else:
                    ls_rollout_rewards.append(reward[0])
                    ls_rollout_penalties.append(penalty)
                    penalties.append(np.asarray([penalty] * config.n_agents))

            episode_rewards.append(np.mean(ls_rollout_rewards))
            episode_penalties.append(np.mean(ls_rollout_penalties))
            penalties = np.asarray(penalties)

            replay_buffer.push(obs, agent_actions, rewards, penalties, next_obs, dones)

            obs = next_obs
            t += config.n_rollout_threads
            lag_mul_ls = []
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')

                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    penalty_q = model.update_penalty_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                    lb_t_ = torch.max(torch.tensor(0.0),
                                        (lb_t + ((penalty_q.mean() - alpha).float() * b_t)))

                    lag_mul_ls.append(lb_t_)
                    # logger.add_scalar('penalty_q', penalty_q.mean(), ep_i)

                # Lagrange Parameter for next ts is the mean over lag parameters computed for all num_updates
                lb_t = torch.from_numpy(np.asarray(lag_mul_ls)).mean()

                model.prep_rollouts(device='cpu')

        # Total costs and penalties over an entire episode
        all_rewards.append(np.sum(episode_rewards))
        all_penalties.append(np.sum(episode_penalties))

        # Mean of total costs and penalties over latest 1024 episodes

        log_rew = np.mean(all_rewards[-1024:])
        log_penalty1 = np.mean(all_penalties[-1024:])

        logger.add_scalar("Mean cost over latest 1024 epi/Training:-", log_rew, ep_i)
        logger.add_scalar("Mean penalty_1 over latest 1024 epi/Training:-", log_penalty1, ep_i)
        logger.add_scalar('lbt', lb_t, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            model.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            model.save(str(run_dir / 'model.pt'))

    model.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_spread', help="Name of environment")
    parser.add_argument("--model_name", default='Cooperative Navigation',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--b_t", default=0.00001, type=int, help="Slower Timescale")
    parser.add_argument("--alpha", default=3, type=int, help="Penalty Threshold on Agents")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=200000, type=int)
    parser.add_argument("--n_agents", default=5, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=50, type=int)
    parser.add_argument("--num_updates", default=2, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", default=True, action='store_true')
    config = parser.parse_args()

    run(config)
