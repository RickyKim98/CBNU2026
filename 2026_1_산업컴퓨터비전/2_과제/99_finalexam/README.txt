Mosquito Glue Tape Analyzer - Toolbar Open Version

실행 방법
1. main.py를 실행합니다.
2. 프로그램 시작 시 흰색 빈 Figure 창만 표시됩니다.
3. 상단 toolbar의 "📂 열기" 버튼을 누릅니다.
4. 분석할 모기 이미지를 선택합니다.
5. 분석이 끝나면 같은 창에 overlay 결과가 표시되고, 콘솔에는 종류별 개체 수가 출력됩니다.
6. ESC 또는 Ctrl+O를 누르면 다시 파일 선택창이 열립니다.

출력 위치
- outputs/overlay : 분석 결과 overlay 이미지
- outputs/crops : 개체별 crop 이미지
- outputs/masks : 중간 mask 이미지
- outputs/reports : summary.txt

Mosquito Glue Tape Analyzer - Improved Accuracy Version
======================================================

목적
----
Glue tape 포획 이미지에서 모기 후보를 검출하고, 각 개체 옆에
한국숲모기 / 갈색모기 / Unknown을 표시하며 콘솔에 종류별 개체 수를 출력합니다.

개선된 알고리즘
---------------
1. Hough Circle + circular contour 기반 glue tape 구멍 검출
2. 구멍 mask를 크게 확장하여 구멍 테두리 오검출 감소
3. K-means segmentation: LAB 색상 + XY 위치 기반으로 파란 배경/모기 후보 분리
4. HSV + LAB 색상 특징: 검정/갈색 분류 안정화
5. Black-hat morphology: 밝은 배경 위 어두운 몸통 후보 보강
6. Contour / Connected Component + Shape Feature: 후보 필터링
7. HOG Feature: 최종 분류보다는 모기 형태 검증 보조로 사용

실행 방법
---------
1. 필요한 패키지 설치
   pip install -r requirements.txt

2. 실행
   python main.py

3. 파일 선택창에서 sample_images 폴더의 이미지 또는 사용자의 이미지를 선택합니다.

출력
----
outputs/overlay : 종류와 ID가 표시된 결과 이미지
outputs/crops   : 검출된 모기 후보 crop 이미지
outputs/masks   : hole mask, candidate mask, kmeans/color/blackhat mask
outputs/reports : summary.txt

주의
----
- 이 프로그램은 생물학적 최종 동정기가 아니라 수업 프로젝트용 종 분류 보조 프로그램입니다.
- 겹친 모기, 훼손 개체, 반사광이 심한 영역은 Unknown/Review로 처리될 수 있습니다.
- 실제 정확도는 촬영 조명, 초점, glue tape 오염 정도에 영향을 받습니다.

[ESC 재로딩 기능]
- main.py 실행 후 이미지를 선택해 분석합니다.
- 결과 창에서 ESC를 누르면 결과 창이 닫히고 파일 선택창이 다시 열립니다.
- ESC가 아닌 다른 키를 누르면 프로그램이 종료됩니다.

[수정 사항]
- 구멍과 겹친 후보를 무조건 제거하지 않습니다.
- 후보 mask에서 hole mask를 뺀 뒤 남는 픽셀이 거의 없을 때만 구멍 음영 오검출로 판단하여 제외합니다.
- 모기가 구멍 위에 걸쳐 있어도 몸통/다리 픽셀이 남으면 후보로 유지합니다.

[2026-06 크기 측정 및 오른쪽 Index Panel 추가]
- 포획 테이프 홀 실제 지름 5.00mm를 고정 기준자로 사용합니다. 이 값은 추정값이 아닙니다.
- 검출된 홀들의 이미지상 지름(px) 중앙값으로 pixel/mm 환산계수를 계산합니다.
- 각 후보의 긴 축 길이를 실제 홀 5.00mm 기준으로 mm 단위 환산하여 overlay 라벨에 표시합니다.
- 결과 이미지 오른쪽 흰색 패널에 ID별 종류와 크기(mm)를 index 형태로 표시합니다.
- 설정 위치: config.py의 hole_diameter_mm = 5.0  # 실제 기준 홀 지름(mm)


[측정값 표기 기준]
- 홀 지름 5.00mm는 실제 고정 기준값입니다.
- 이미지에서 검출된 홀 지름(px)을 측정하여 pixel/mm 환산계수를 구합니다.
- 오른쪽 Index Panel의 mm 값은 이 환산계수를 이용한 검출영역 긴 축 길이입니다.
- 다리/날개 포함 여부, 모기 자세, 겹침 상태에 따라 생물학적 몸길이와는 차이가 날 수 있습니다.
