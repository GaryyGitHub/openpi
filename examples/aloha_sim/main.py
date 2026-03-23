import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import eval_stats as _eval_stats
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False

    num_episodes: int = 25  # 固定为 25 个 episode
    max_episode_steps: int = 0


def main(args: Args) -> None:
    environment = _env.AlohaSimEnvironment(
        task=args.task,
        seed=args.seed,
    )

    stats = _eval_stats.EvalStats(environment, max_score=4.0)  # 设置满分为 4 分

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
            stats,
        ],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()

    print(stats.summary())  # 输出得分率统计


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
