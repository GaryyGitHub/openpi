from openpi_client.runtime import subscriber as _subscriber


class EvalStats(_subscriber.Subscriber):
    def __init__(self, environment, max_score: float = 4.0) -> None:
        self._env = environment
        self._max_score = max_score
        self.total_score = 0.0
        self.total_count = 0
        self.episode_scores = []  # 新增: 记录每个 episode 的得分

    def on_episode_start(self) -> None:
        pass

    def on_step(self, observation: dict, action: dict) -> None:
        pass

    def on_episode_end(self) -> None:
        self.total_count += 1
        score = 0.0
        if hasattr(self._env, "get_episode_score"):
            score = self._env.get_episode_score()  # type: ignore
        self.total_score += score
        self.episode_scores.append(score)  # 保存得分

        print(f"Episode {self.total_count}: Score={score:.4f} | Total Score: {self.total_score:.4f}")

    def average_score_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_score / (self.total_count * self._max_score)

    def summary(self) -> str:
        episode_scores_str = ", ".join(f"{score:.4f}" for score in self.episode_scores)  # 格式化得分列表
        return f"Average Score Rate: {self.average_score_rate():.2%} ({self.total_score:.4f}/{self.total_count * self._max_score:.4f})\nEpisode Scores: [{episode_scores_str}]"
