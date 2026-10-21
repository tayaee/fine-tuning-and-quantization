#!/usr/bin/env python3
"""report_check.py — report_to 검증 단일 진실 공급원.

check_report_to.py(사전 점검)와 train-entry.py(기동 게이트)가 공유한다.
검증은 Trainer가 실제로 타는 경로와 동일한 코드로 수행한다:
  transformers.integrations.integration_utils.get_reporting_integration_callbacks
  + 콜백 인스턴스화(Trainer.__init__의 `cb() if isinstance(cb, type)` 행과 동일).
explicit 미설치 백엔드는 Trainer에서도 인스턴스화 시점에 ValueError로 죽으므로
그대로 재현된다. setup() 시점 폭사(mlflow file-store)는 임시 디렉토리로 모사 검증한다.

모듈 import 시에는 stdlib만 사용한다 (transformers는 검증 함수 안에서 import).
"""

import json
import os
import subprocess
import sys

try:
    import importlib.metadata as _metadata
except ImportError:  # pragma: no cover
    _metadata = None

# transformers INTEGRATION_TO_CALLBACK 키(5.x)와 1:1 대응. 표시용 메타.
# KNOWN을 줄이면 체커의 all이 Trainer의 all과 어긋나므로 14종 전체 유지.
KNOWN_META = {
    "tensorboard": {
        "pip": "tensorboard",
        "env": ["TENSORBOARD_LOGGING_DIR"],
        "out": lambda out: os.getenv("TENSORBOARD_LOGGING_DIR")
        or (os.path.join(out, "runs") if out else "$OUT/runs"),
        "note": "로컬 이벤트 파일. rank0만 기록.",
    },
    "mlflow": {
        "pip": "mlflow",
        "import": "mlflow",
        "env": ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_ALLOW_FILE_STORE"],
        "out": lambda out: os.getenv("MLFLOW_TRACKING_URI") or "./mlruns",
        "note": "file-store는 maintenance 모드. ALLOW_FILE_STORE 없으면 setup 폭사.",
    },
    "trackio": {
        "pip": "trackio>=0.21.1",
        "env": [],
        "out": lambda out: os.path.join(
            os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
            "trackio/",
            "trackio-math-1b-sft-and-quantization.db",
        ),
        "note": "로그인 불필요(로컬). rank0만 기록.",
    },
    "dvclive": {
        "pip": "dvclive",
        "env": ["HF_DVCLIVE_LOG_MODEL"],
        "out": lambda out: "./dvclive/ (metrics.tsv, params.yaml)",
        "note": "cwd 기준. rank0만 기록.",
    },
    "wandb": {
        "pip": "wandb",
        "env": ["WANDB_PROJECT", "WANDB_MODE", "WANDB_API_KEY"],
        "out": lambda out: f"cloud wandb.ai/{os.getenv('WANDB_PROJECT', 'wandb-math-1b-sft-and-quantization')} + ./wandb/",
        "note": "유일하게 인증 필요(wandb login). offline이면 WANDB_MODE=offline.",
    },
    "comet_ml": {
        "pip": "comet-ml",
        "env": ["COMET_API_KEY", "COMET_PROJECT_NAME"],
        "out": lambda out: "comet.com 클라우드",
        "note": "SaaS. API 키 필요.",
    },
    "swanlab": {
        "pip": "swanlab",
        "env": ["SWANLAB_API_KEY"],
        "out": lambda out: "swanlab 클라우드 + ./swanlog/",
        "note": "SaaS.",
    },
    "clearml": {
        "pip": "clearml",
        "env": [],
        "out": lambda out: "ClearML 서버(셀프호스팅/클라우드)",
        "note": "MLOps 확장형. 무거움.",
    },
    "dagshub": {
        "pip": "dagshub",
        "env": ["MLFLOW_TRACKING_URI"],
        "out": lambda out: "DagsHub 원격 (MLFLOW_TRACKING_URI로 지정)",
        "note": "DagsHub 계정 필요.",
    },
    "neptune": {
        "pip": "neptune",
        "env": ["NEPTUNE_API_TOKEN", "NEPTUNE_PROJECT"],
        "out": lambda out: "neptune.ai 클라우드",
        "note": "SaaS. API 토큰 필요.",
    },
    "codecarbon": {
        "pip": "codecarbon",
        "env": [],
        "out": lambda out: "./codecarbon / emissions.csv",
        "note": "실험추적 아님(탄소 측정). loss용 아님.",
    },
    "azure_ml": {
        "pip": "azureml-sdk",
        "env": [],
        "out": lambda out: "Azure ML workspace run history",
        "note": "Azure 환경 종속.",
    },
    "flyte": {
        "pip": "flytekit",
        "env": [],
        "out": lambda out: "Flyte deck / K8s 파이프라인",
        "note": "단일 학습 로그용 아님.",
    },
    "kubeflow": {
        "pip": "kfp",
        "env": [],
        "out": lambda out: "Kubeflow Pipelines UI (자동 추가)",
        "note": "설치돼 있으면 transformers가 자동 추가.",
    },
}

STATIC_KEYS = sorted(KNOWN_META.keys())

# transformers가 요구하는 최소 버전 (integration_utils 상수에서 발췌).
MIN_VERSIONS = {
    "trackio": "0.21.1",  # TrackioCallback.MIN_TRACKIO_VERSION_FOR_FREEZE
}

DEEP_TIMEOUT = 300

# 자식 프로세스 검증 스크립트. Trainer와 동일한 해석+인스턴스화를 수행한다.
# stdout 마지막 줄에 JSON 배열 출력: [{name, status, detail}].
# status: INST-OK | MISSING | BROKEN | UNKNOWN | EMPTY
_CHILD_SCRIPT = r"""
import sys, json, os
req = sys.argv[1:]
from transformers.integrations.integration_utils import (
    get_available_reporting_integrations,
    get_reporting_integration_callbacks,
)
rows = []
def check_one(n):
    try:
        cbs = get_reporting_integration_callbacks(n)
    except ValueError as e:
        rows.append({"name": n, "status": "UNKNOWN",
                     "detail": (str(e).splitlines() or ["unsupported"])[0][:300]})
        return
    if not cbs:
        rows.append({"name": n, "status": "EMPTY", "detail": "no callbacks"})
        return
    for cb in cbs:
        cname = getattr(cb, "__name__", type(cb).__name__)
        try:
            inst = cb() if isinstance(cb, type) else cb
        except Exception as e:
            s = str(e) or ""
            msg = (s.splitlines() or [""])[0][:300]
            st = "MISSING" if ("to be installed" in s or "pip install" in s) else "BROKEN"
            rows.append({"name": n, "status": st,
                         "detail": "%s: %s: %s" % (cname, type(e).__name__, msg)})
            return
        rows.append({"name": n, "status": "INST-OK", "detail": cname})
    if n == "mlflow" and any(r["name"] == "mlflow" and r["status"] == "INST-OK" for r in rows):
        import tempfile
        d = tempfile.mkdtemp(prefix="m1b-mlflow-probe-")
        os.environ["MLFLOW_TRACKING_URI"] = "file:" + d
        try:
            from mlflow.tracking import fluent
            fluent.set_experiment("preflight-check")
        except Exception as e:
            s = str(e) or ""
            msg = (s.splitlines() or [""])[0][:300]
            for r in rows:
                if r["name"] == "mlflow" and r["status"] == "INST-OK":
                    r["status"] = "BROKEN"
                    r["detail"] = "setup(file-store) 실패: %s: %s" % (type(e).__name__, msg)
for n in req:
    if n == "all":
        inst = sorted(get_available_reporting_integrations())
        if not inst:
            rows.append({"name": "all", "status": "EMPTY", "detail": "installed none"})
        for i in inst:
            check_one(i)
    else:
        check_one(n)
print(json.dumps(rows))
"""


def parse_report_to(s):
    """train-entry.py와 동일한 파싱. SFTConfig에 넘어가는 값을 반환."""
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    if not items or items == ["none"]:
        return "none"
    if len(items) == 1:
        return items[0]
    filtered = [x for x in items if x != "none"]
    return filtered if filtered else "none"


def _version_ok(name):
    """MIN_VERSIONS floor 검사. (True, '') 또는 (False, 이유)."""
    if name not in MIN_VERSIONS or _metadata is None:
        return True, ""
    try:
        from packaging.version import Version
    except ImportError:
        return True, ""
    try:
        ver = _metadata.version(KNOWN_META[name]["pip"].split(">=")[0].split("==")[0])
    except Exception:
        return True, ""
    if Version(ver) < Version(MIN_VERSIONS[name]):
        return False, f"버전 미달: {ver} < 최소 {MIN_VERSIONS[name]}"
    return True, ""


def summarize(value, out=None):
    """검증 수행. 반환: {"effective", "rows": [{name,status,detail,output,env,note}], "problems"}.
    status 최종값: OK | MISSING | BROKEN | UNKNOWN | ENV-BROKEN.
    """
    effective = parse_report_to(value)
    if effective == "none" or effective == []:
        return {"effective": "none", "rows": [], "problems": 0}
    argv = list(effective) if isinstance(effective, list) else [effective]
    try:
        r = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT] + argv,
            capture_output=True, text=True, timeout=DEEP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"effective": effective, "rows": [{
            "name": ",".join(argv), "status": "ENV-BROKEN",
            "detail": f"검증 서브프로세스 타임아웃({DEEP_TIMEOUT}s)",
            "output": "", "env": "", "note": ""}], "problems": 1}
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or "")).strip().splitlines()[-5:]
        return {"effective": effective, "rows": [{
            "name": ",".join(argv), "status": "ENV-BROKEN",
            "detail": "transformers 검증 경로 실행 실패: " + " / ".join(t.strip() for t in tail),
            "output": "", "env": "", "note": ""}], "problems": 1}
    try:
        child_rows = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception as e:
        return {"effective": effective, "rows": [{
            "name": ",".join(argv), "status": "ENV-BROKEN",
            "detail": f"검증 결과 파싱 실패: {e}",
            "output": "", "env": "", "note": ""}], "problems": 1}

    rows = []
    for cr in child_rows:
        n, st, detail = cr["name"], cr["status"], cr.get("detail", "")
        if st == "INST-OK":
            ok, why = _version_ok(n)
            if not ok:
                st, detail = "BROKEN", why
        meta = KNOWN_META.get(n, {"pip": n, "env": [], "out": lambda o: "(벤더 문서 참조)", "note": ""})
        try:
            outp = meta["out"](out)
        except Exception:
            outp = "(경로 확인 실패)"
        env_show = ""
        if meta.get("env"):
            env_show = " | " + ", ".join(f"{k}={os.getenv(k) or '(미설정)'}" for k in meta["env"])
        if st == "INST-OK":
            st = "OK"
        if n == "wandb" and st == "OK" and not os.getenv("WANDB_API_KEY"):
            detail += " | HINT: WANDB_API_KEY 미설정 — 'wandb login' 확인 필요"
        rows.append({"name": n, "status": st, "detail": detail,
                     "output": outp, "env": env_show,
                     "note": meta.get("note", ""), "pip": meta.get("pip", n)})
    problems = sum(1 for x in rows if x["status"] != "OK")
    return {"effective": effective, "rows": rows, "problems": problems}


def gate_report_to(value):
    """train-entry.py 기동 게이트. 문제 있으면 stderr 출력 후 SystemExit(2).
    dry_run 뒤·무거운 작업 전에 호출할 것."""
    res = summarize(value)
    if res["problems"]:
        print(f"[report-gate] REFUSE: report_to={value!r} -> {res['effective']!r}", file=sys.stderr)
        for x in res["rows"]:
            if x["status"] != "OK":
                print(f"[report-gate] [{x['status']}] {x['name']}: {x['detail']}", file=sys.stderr)
        raise SystemExit(2)
