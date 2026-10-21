"""distributed/ray-10-baseline.py — L2. Ray Train으로 Stage 1 동일 학습을 분산 실행.
workers = 1 (1x) | 2 (2x). TorchTrainer가 train-entry.py의 train_fn을 감싼다.

  uv run distributed/ray-10-baseline.py --mode mini --infra dgx-spark-1x|dgx-spark-2x
전제: spark1·spark2에 동일 repo 경로 + ray 클러스터 (ray start --head / --address).
"""

import argparse


def train_fn(config: dict):
    # TODO: train-entry.py 로직을 함수로 승격 후 여기서 호출
    # (storage_path=$M1B_SHARED/models/base-gsm8k-<mode> — 양 노드 공통 경로 필수)
    print(f"STUB train_fn config={config}")


def main(mode: str, infra: str):
    workers = 2 if infra == "dgx-spark-2x" else 1
    # TODO: from ray.train.torch import TorchTrainer; TorchTrainer(train_fn, ...).fit()
    print(f"[{mode}][{infra}] workers={workers}")
    print("STUB: Ray Trainer body not implemented")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument(
        "--infra", default="dgx-spark-1x", choices=["dgx-spark-1x", "dgx-spark-2x"]
    )
    a = ap.parse_args()
    main(a.mode, a.infra)
