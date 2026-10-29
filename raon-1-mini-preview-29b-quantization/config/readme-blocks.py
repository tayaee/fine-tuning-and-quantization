"""config/readme-blocks.py — 모델 카드(README.md)의 코드 블록을 바이트 그대로 꺼낸다.

s6/s7 "README as-is" 검증 공용. 고객이 보는 Hub README에서 `## <section>` 아래
fenced code block들을 순서대로 추출해 <outdir>/block-<i>.<lang> 로 **원본 바이트 그대로**
쓰고, snippets/<target>/ 의 대응 스니펫(__REPO_ID__ 치환)과 바이트 비교한다.

바이트 정확성 규칙:
  - README는 bytes로 읽는다 (universal newline 변환 금지). CRLF가 섞인 카드도
    원본 그대로 재현해야 하므로 text 모드 변환을 하지 않는다.
  - 스니펫도 bytes로 읽어 비교한다. 개행 정규화 차이가 DIFF로 잡히지 않는다.

사용:
  python readme-blocks.py <README.md> "<section>" <outdir> <repo> \
      [--expect-no-blocks|--expect-blocks] [<snippet-path-or-dash> ...]
  - 위치 인자 snippet i 는 block i 와 대응. '-' 는 비교 생략 (스니펫 없는 블록).
  - --expect-no-blocks: 이 섹션에 코드 블록이 없어야 한다 (vLLM 전용 카드의
    '## Usage - transformers' 가 실행 예제를 제시하지 않는지 확인). 있으면 rc=1.
  - --expect-blocks: 이 섹션에 코드 블록이 있어야 한다 (실행 예제가 있는 카드). 없으면 rc=1.
  - 출력: block 파일 목록, IDENTICAL/DIFF 판정. 하나라도 DIFF면 rc=1, 섹션/블록 없음 rc=2.
  - <outdir>/blocks.json 에 {section, readme_sha256, blocks:[{file, lang, sha256, verdict}]} 기록.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

# `## <title>` 섹션: 다음 `## ` 직전까지. 제목은 정규식 이스케이프.
# bytes 대상이므로 패턴도 bytes로 (str 패턴은 bytes에 쓸 수 없다).
_SECTION_HEAD = rb"^## "
_SECTION_TAIL = rb"[ \t]*\r?\n(.*?)(?=^## |\Z)"
# fenced block: ```lang\n ... \n``` (닫는 fence는 줄 시작)
_BLOCK_RE = rb"^```([\w+-]*)[ \t]*\r?\n(.*?)^```[ \t]*\r?$"


def section_body(text: bytes, title: str) -> bytes | None:
    pat = _SECTION_HEAD + re.escape(title.encode()) + _SECTION_TAIL
    m = re.search(pat, text, re.S | re.M)
    return m.group(1) if m else None


def code_blocks(body: bytes) -> list[tuple[bytes, bytes]]:
    return [(lang, code) for lang, code in
            re.findall(_BLOCK_RE, body, re.S | re.M)]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def first_diff(want: bytes, got: bytes) -> str:
    """DIFF 판정 사유를 사람이 읽을 수 있게 1줄로 요약한다."""
    w = want.decode("utf-8", errors="replace").splitlines()
    g = got.decode("utf-8", errors="replace").splitlines()
    for i, (a, b) in enumerate(zip(w, g), 1):
        if a != b:
            return f"line {i}: snippet={a!r} readme={b!r}"
    if len(w) != len(g):
        return f"line count: snippet={len(w)} readme={len(g)}"
    return "identical after decode (byte difference only)"


def show_diff(want: bytes, got: bytes, limit: int = 20) -> str:
    w = want.decode("utf-8", errors="replace").splitlines(keepends=True)
    g = got.decode("utf-8", errors="replace").splitlines(keepends=True)
    out = list(difflib.unified_diff(w, g, "snippet", "readme", n=1))
    return "".join(out[:limit])


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    args = argv[1:]
    expect_no_blocks = False
    expect_blocks = False
    if "--expect-no-blocks" in args:
        expect_no_blocks = True
        args.remove("--expect-no-blocks")
    if "--expect-blocks" in args:
        expect_blocks = True
        args.remove("--expect-blocks")
    if len(args) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    readme, title, outdir, repo, snippets = args[0], args[1], Path(args[2]), args[3], args[4:]
    text = Path(readme).read_bytes()
    body = section_body(text, title)
    if body is None:
        print(f"FAIL: README에 '## {title}' 섹션 없음", file=sys.stderr)
        return 2
    blocks = code_blocks(body)
    outdir.mkdir(parents=True, exist_ok=True)
    ext = {"bash": "sh", "sh": "sh", "python": "py"}
    meta = {"section": title, "readme_sha256": sha(text), "blocks": []}
    if expect_no_blocks:
        (outdir / "blocks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
        if blocks:
            print(f"FAIL: '## {title}' 에 코드 블록 {len(blocks)}개가 있음 — "
                  "이 타깃은 transformers로 적재 불가하므로 카드가 실행 예제를 제시하면 안 된다",
                  file=sys.stderr)
            for i, (lang_b, code) in enumerate(blocks):
                print(f"  block-{i} [{lang_b.decode('ascii', 'replace')}] {len(code)}B", file=sys.stderr)
            return 1
        print(f"OK: '## {title}' 에 코드 블록 없음 (vLLM 전용 카드)")
        return 0
    if expect_blocks and not blocks:
        print(f"FAIL: '## {title}' 에 코드 블록이 없다 — 카드에 실행 예제가 있어야 한다",
              file=sys.stderr)
        return 1
    if len(blocks) < len(snippets):
        print(f"FAIL: '## {title}' 블록 {len(blocks)}개 < 기대 {len(snippets)}개", file=sys.stderr)
        return 2
    bad = 0
    for i, (lang_b, code) in enumerate(blocks):
        lang = lang_b.decode("ascii", errors="replace")
        f = outdir / f"block-{i}.{ext.get(lang, lang or 'txt')}"
        f.write_bytes(code)  # 원본 바이트 그대로
        verdict = "-"
        if i < len(snippets) and snippets[i] != "-":
            want = Path(snippets[i]).read_bytes().replace(b"__REPO_ID__", repo.encode())
            verdict = "IDENTICAL" if want == code else "DIFF"
            if verdict == "DIFF":
                bad += 1
                print(f"  {first_diff(want, code)}", file=sys.stderr)
                print(show_diff(want, code), file=sys.stderr)
        meta["blocks"].append({"file": f.name, "lang": lang, "sha256": sha(code),
                               "snippet": snippets[i] if i < len(snippets) else "-",
                               "verdict": verdict})
        print(f"block-{i} [{lang}] {len(code)}B sha256={sha(code)[:12]} snippet={verdict}")
    (outdir / "blocks.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    if bad:
        print(f"FAIL: README 블록 {bad}개가 snippets/와 다름 (카드 재업로드 필요)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
