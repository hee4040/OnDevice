# GitHub 업로드 방법 (브랜치 생성이 ChatGPT GitHub 연결에서 403으로 거부된 경우)

1. ZIP을 내려받아 압축 해제합니다.
2. GitHub 저장소 `hee4040/OnDevice`에 `push` 권한이 있는 계정으로 Git 인증을 설정합니다 (`gh auth login` 또는 Git Credential Manager).
3. 압축 해제한 폴더에서 **Git Bash**를 열고 `bash push_code_E.sh`를 실행합니다.
4. 스크립트는 `main`을 복제해 `experiment/code_E`라는 **새 브랜치**를 만듭니다. 기존 `main`과 다른 팀 브랜치에는 직접 쓰지 않습니다.
5. `README.md`의 기존 내용을 보존하고 `README_code_E_section.md`를 마지막에 추가하며, `experiments/code_E`의 Notebook·PDF·Word를 커밋하고 push합니다.

주의: GitHub에 같은 이름의 브랜치가 이미 존재하면 덮어쓰지 않고 종료합니다. Notebook은 이전 대화에서 정리했던 원본 셀 순서를 바탕으로 다시 생성한 버전이며 Colab 전체 실행을 실제 재검증한 것은 아닙니다. PDF 역시 이전 보고서의 방법과 주요 수치를 바탕으로 다시 만든 버전입니다.
