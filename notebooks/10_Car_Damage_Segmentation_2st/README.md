# Vehicle 8-View Classifier (Rule-based)

기존에 학습된 **YOLOv8-Seg (차량 파손/부품 탐지)** 모델의 예측 결과를 활용하여, **차량의 촬영 방향(8면)** 을 자동으로 분류

## 1. 개요 (Overview)
- **목표**: 사용자로부터 수집된 차량 이미지 8면(앞, 뒤, 좌, 우, 좌상, 우상, 좌후, 우후)의 정합성을 검증하고 자동 분류
- **방식**: **Rule-based Logic** (YOLO가 탐지한 부품의 조합을 통해 뷰를 역추적)
- **장점**:
  - 별도의 Image Classification 모델 학습 불필요 (비용/시간 절감).
  - 부품 검출 성능이 높다면 분류 정확도 또한 매우 높음.
  - 판단 근거(XAI)가 명확함 (예: "앞범퍼와 왼쪽 문이 동시에 보여서 좌상단으로 판단함").

---

## 2. 알고리즘 로직: 점수 산정 (Scoring System)

YOLO가 탐지한 객체(`Confidence > 0.25`)를 분석하여 4방향 점수(`F`, `B`, `L`, `R`)를 누적

### 부품별 가중치 테이블 (Weight Table)

| 구분 (Category) | 감지된 부품 (Class Name) | 점수 (Score) | 논리 (Logic) |
| :--- | :--- | :--- | :--- |
| **Absolute Front** | `Front bumper` | **F +4** | 범퍼는 정면을 결정짓는 절대적 기준 |
| **Major Front** | `Bonnet` | **F +3** | 본네트는 정면에서 넓게 보임 |
| **Minor Front** | `Windshield`, `Head lights` | **F +1~2** | 헤드라이트(+2), 앞유리(+1) |
| **Absolute Back** | `Rear bumper` | **B +4** | 뒷범퍼는 후면을 결정짓는 절대적 기준 |
| **Major Back** | `Trunk lid` | **B +3** | 트렁크 리드는 후면에서 넓게 보임 |
| **Minor Back** | `Rear windshield`, `Rear lamp` | **B +1~2** | 리어램프(+2), 뒷유리(+1) |
| **Side Bias** | 이름에 `(L)` 포함 | **L +3** | 왼쪽 부품 감지 시 좌측 점수 대폭 상승 |
| **Side Bias** | 이름에 `(R)` 포함 | **R +3** | 오른쪽 부품 감지 시 우측 점수 대폭 상승 |
| **Corner (Hybrid)** | `Front fender`, `Rear fender` | **F/B +2**<br>**L/R +3** | **[핵심]** 휀다는 앞/뒤 점수와 측면 점수를 **동시에** 획득하여 대각선 판정을 유도함 |

---

## 3. 결정 트리 (Decision Logic)

점수가 산정된 후, 아래의 **우선순위(Priority)** 에 따라 최종 뷰를 결정

### Priority 1. 양안 시각 (Binocular Vision)
사람처럼 **"양쪽 눈(라이트)이 다 보이면 정면"** 이라고 판단
- **Condition**: `Head lights(L)` AND `Head lights(R)` 감지됨
- **Result**: **`Front`** (측면 점수가 아무리 높아도 무시)

### Priority 2. 동점 방지 및 정면/후면 확정 (Tie-Breaking)
좌/우 점수가 비슷하다면, 측면이 아니라 정면/후면일 확률이 높습니다. (노이즈 방어)
- **Scenario**: `F=4`, `L=3`, `R=3` (범퍼도 보이고, 좌/우 휀다가 살짝씩 다 보임)
- **Condition**: `F >= 2` AND `abs(L - R) <= 2` (좌우 점수 차이가 2점 이내)
- **Result**: **`Front`** (대각선으로 오분류 방지)

### Priority 3. 대각선 뷰 (Corner View)
앞쪽 점수도 높고, **한쪽** 측면 점수만 확실히 높을 때입니다.
- **Scenario**: `F=4` (범퍼), `L=5` (휀다+도어), `R=0`
- **Condition**: `F >= 2` AND `L >= 2` AND `L > R` (왼쪽이 오른쪽보다 확실히 커야 함)
- **Result**: **`Front-Left`**

### Priority 4. 완전 측면 (Pure Side)
앞/뒤 점수는 거의 없고 측면 점수만 높을 때입니다.
- **Scenario**: `F=0`, `L=6` (도어 2개), `R=0`
- **Condition**: `L > F` AND `L > B`
- **Result**: **`Left`**

---

## 4. 로직 시뮬레이션 예시 (Case Study)


#### Case A: 명확한 대각선 (Front-Left)
- **Detected**: `Front bumper`, `Front fender(L)`, `Head lights(L)`
- **Scoring**:
  - Bumper: `F+=4`
  - Fender(L): `F+=2`, `L+=3` (양면성 적용)
  - Light(L): `F+=2`
- **Total**: **`F=8`, `L=3`, `R=0`**
- **Decision**: `F`가 높고, `L`도 임계값(2) 이상이며, `L > R` 이므로 → **`Front-Left`**

#### Case B: 애매한 각도/노이즈 방어 (Tie-Breaking)
- **Detected**: `Front bumper`, `Front fender(R)`, `Side mirror(L)` (노이즈 발생)
- **Scoring**:
  - Bumper: `F+=4`
  - Fender(R): `F+=2`, `R+=3`
  - Mirror(L): `L+=2`
- **Total**: **`F=6`, `L=2`, `R=3`**
- **Decision**:
  - `L`과 `R`의 차이가 `1` (<= 2) 이므로 측면 편향이 적음.
  - `F`가 충분히 높음.
  - → **`Front`** (우상단으로 잘못 빠지지 않음)

#### Case C: 완벽한 후면 (Pure Back)
- **Detected**: `Rear bumper`, `Trunk lid`, `Rear lamp(L)`, `Rear lamp(R)`
- **Decision**: 양쪽 `Rear lamp`가 모두 감지됨 (Binocular Rule)
- → 점수 계산할 필요 없이 즉시 **`Back`** 확정.

---
| sample |
| :---: |
| <img src="./results/images10.png" width="50%"> |
| <img src="./results/images20.png" width="50%"> |
| <img src="./results/images30.png" width="50%"> |
| <img src="./results/images40.png" width="50%"> |
| <img src="./results/images50.png" width="50%"> |

