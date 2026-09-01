# Basketball Shot Biomechanics Evidence Base

Date: 2026-09-01

## Evidence policy

- A study association is not a universal coaching rule and is not proof of causality.
- Population, shot type, distance, defense, fatigue and camera view must travel with every metric.
- Product output distinguishes measured value, image-plane proxy, model estimate and unavailable.
- Ordinary monocular phone video does not support force, torque, joint loading, true 3D kinetic chain or injury-risk claims without validation.
- Recommendations default to within-person consistency and matched-condition comparison, not population means.

## Evidence sources

1. Okazaki, Rodacki and Satern, 2015 systematic review: https://pubmed.ncbi.nlm.nih.gov/26102462/
2. Youth jump-shot systematic review, 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8005190/
3. Miller and Bartlett, distance/position, 1996: https://pubmed.ncbi.nlm.nih.gov/8809716/
4. Rojas et al., defender condition, 2000: https://pubmed.ncbi.nlm.nih.gov/11083144/
5. Increased shooting distance, 2013: https://pubmed.ncbi.nlm.nih.gov/24149195/
6. Female adolescent distance study, 2024: https://pubmed.ncbi.nlm.nih.gov/38314460/
7. Near-minimum release-speed strategy, 2020: https://pubmed.ncbi.nlm.nih.gov/32217201/
8. Hierarchical redundancy/coordination, 2022: https://pubmed.ncbi.nlm.nih.gov/34968877/
9. Misses versus swishes coordination variability, 2010: https://pubmed.ncbi.nlm.nih.gov/20552519/
10. Repeated-sprint fatigue/three-point shooting, 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC6006537/
11. Context constraints and accuracy, 2025: https://pubmed.ncbi.nlm.nih.gov/41283566/
12. Athlete-versus-novice free throws, 2026: https://pubmed.ncbi.nlm.nih.gov/42358510/

## Product metric evidence matrix

| Metric | Evidence | Population / shot / condition | Observed effect | Strength | Monocular phone observability | Expected error / view dependency | Safe recommendation? | Allowed wording | Forbidden claim |
|---|---|---|---|---|---|---|---|---|---|
| Strict release frame | Ball trajectory studies require a release instant; HOI literature defines contact transition. | Across studies, usually controlled free throw/jump shot. | Release anchors angle, height, speed and phase timing. | Strong task definition; weak current project detector validation. | Potentially observable with ball track + hand/contact at adequate FPS/resolution. | At 30 FPS, one frame is 33.3 ms; hand occlusion can add multiple frames. Side/diagonal view preferred. | Yes, as evidence/uncertainty. | “Ball-hand separation is first supported at frame X, tolerance +/- N frames.” | “The exact biological release occurred at X” when contact is occluded. |
| Release height | 2013 distance study; 2024 adolescent study; 2026 free-throw study. | Small expert/youth/adult samples; distance and free throw differ. | Often decreases with distance; higher within-protocol release height associated with success in 2026 free throws. | Moderate, context-dependent. | Image-plane relative height is observable; metric metres require scale/calibration. | Perspective and out-of-plane motion dominate. | Yes, relative to own matched attempts. | “Your release point appears lower than your own recent matched attempts.” | “Everyone should release at 2.28 m.” |
| Release angle | 1996/2013/2024 distance studies; 2020 minimum-speed strategy. | Mixed adult/youth, different distances. | Generally lower at longer distances; optimum depends on release height/speed/distance. | Moderate for contextual adaptation, weak for universal target. | Ball trajectory angle can be estimated only with stable track/camera; hand/forearm angle is not ball release angle. | Sensitive to calibration, lens/perspective and short post-release track. | Conditional only. | “Estimated image-plane trajectory is flatter than your matched close-range attempts.” | “Your release angle must be 52 degrees.” |
| Release velocity | 1996/2013/2024 distance and 2020 strategy studies. | Controlled known distances with high-speed/3D capture. | Velocity rises with distance; experts may choose near-minimum-speed solutions. | Moderate/strong in laboratory context. | Pixel velocity observable; m/s unavailable without scale/calibration/depth. | FPS, blur, camera motion and perspective produce large errors. | Not as metric m/s in normal V1. | “Post-release image displacement per frame is higher in this matched view.” | “Ball speed is X m/s” from uncalibrated phone video. |
| Elbow extension at release | 2026 sagittal free-throw study; 2010 elbow-wrist coordination. | 50 adults free throws; collegiate free throws. | Greater extension associated with successful shots in that protocol; timing variability differed in misses. | Moderate for free throw; limited external validity. | 2D sagittal angle observable when elbow/wrist/shoulder are visible. | Out-of-plane rotation and pose jitter; expected model error must be benchmarked. | Yes, within-person/matched view. | “The shooting elbow appears less extended than in your accepted free throws.” | “This elbow angle caused the miss.” |
| Shoulder flexion/timing | 1996 and 2024 distance studies; defender study. | Distance/defense-specific samples. | Shoulder velocity/flexion adapts to distance and defender. | Moderate contextual evidence. | 2D side-view proxy feasible. | Poor in frontal/rear view; clothing/occlusion. | Conditional. | “Shoulder timing differs from your matched attempts under this view.” | “More shoulder flexion is always better.” |
| Wrist/finger action | 2010 coordination and 2024 finger proprioception work; 2026 free throws. | Experienced/college samples, controlled tasks. | Distal coordination contributes to release reproducibility; evidence does not yield a universal visible wrist angle. | Low-to-moderate for product measurement. | Wrist may be visible; fingers usually too small/occluded in 480p. | High uncertainty and hand-model failure around the ball. | Only when hand evidence passes quality gate. | “Wrist/finger detail is unavailable” or “follow-through direction is repeatable across these clips.” | “Finger pressure/spin is measured from ordinary RGB.” |
| Knee flexion / dip depth | Youth review, distance and fatigue studies. | Youth/adult; close/long, fatigued/non-fatigued. | Deeper knee flexion can accompany longer distance/fatigue; not an isolated accuracy determinant. | Moderate contextual, weak causal. | Side-view 2D angle and normalized hip displacement feasible. | Foot/hip visibility and camera tilt matter. | Yes, as strategy/consistency. | “Dip depth increased as distance increased in your matched clips.” | “Your knees must reach a fixed angle.” |
| Jump height / release relative to jump peak | 2024 adolescent distance and 2025 context study. | Adolescent females and 18 senior players under controlled capture. | Release can occur before jump peak; successful attempts sometimes had higher jump/flight/release height, but isolated associations were weak. | Moderate descriptive, weak causal. | Relative vertical displacement and event ordering feasible in fixed view. | True height requires scale/depth; pose jitter affects peak. | Yes, relative timing only. | “Release occurs approximately N frames before your estimated body apex.” | “Releasing at apex is always correct.” |
| Trunk lean / body alignment | Reviews, defender and 2026 free-throw study. | Shot/view/population specific. | Trunk/posture supports adaptation/stability; no single universal value. | Low-to-moderate. | 2D torso axis feasible in side/front view. | Camera roll and out-of-plane turn confound angle. | Yes, with view caveat. | “Torso alignment is more variable across these matched attempts.” | “This trunk lean predicts injury or causes misses.” |
| Landing balance | Phase-specific skill studies and general sports biomechanics; direct cited shooting evidence is thinner. | Jump/stride-stop shots, mostly controlled or recent studies. | Higher skill may show different synchronization/stabilization, but shot type matters. | Preliminary/moderate. | Foot locations, landing interval and normalized sway may be observed if landing remains in frame. | Foot pose, floor plane and occlusion. | Yes, descriptive only. | “Landing is asymmetric relative to take-off in this camera plane.” | “This landing is unsafe” or injury diagnosis. |
| Inter-joint coordination | 2010 and 2022 free-throw studies. | Experienced/collegiate free throws. | Covariation can stabilize release; misses may show late coordination variability. | Moderate mechanistic evidence. | Temporal angle coupling across repeated trials can be estimated. | Derivatives amplify pose/FPS noise; requires repeated attempts. | Yes, as longitudinal consistency. | “Elbow-wrist timing is less consistent across these attempts.” | “One shot proves a broken kinetic chain.” |
| Attempt-to-attempt consistency | 2020 minimum-error strategy, 2022 redundancy, 2010 variability. | Experienced free throws in controlled settings. | Reproducibility and coordinated variability are central to performance. | Strong rationale, metric choice still needs validation. | Highly feasible with repeated matched videos. | Requires matched shot type/view/distance and robust alignment. | Yes; highest-priority V1 quality concept. | “Your release timing varied by N frames across five comparable shots.” | “Population average proves your personal optimum.” |
| Fatigue-related change | Youth review, 2018 repeated-sprint study and other mixed findings. | Elite/youth/male/female protocols differ. | Some studies show changed accuracy/consistency/angles; others show stable kinematics. | Mixed. | Repeated-session changes observable, fatigue itself is not. | Confounded by distance, order, motivation and camera. | No direct fatigue diagnosis. | “Later attempts differ from earlier attempts; fatigue is one possible context to record.” | “The video proves you are fatigued.” |
| Defender/context adaptation | 2000 defender study; 2025 context study. | Professional/senior players under simulated or actual opposition. | Defender can shorten release and alter posture; newer study found weak isolated accuracy associations. | Moderate for adaptation, weak for causal scoring. | Defender presence and relative timing observable. | Player occlusion and varied distance. | Yes, compare like with like. | “This defended shot uses a quicker/higher release than your open attempts.” | “The open-shot template is the correct target under defense.” |

## Measurement classes for V1

### Class A: reasonably supportable from controlled monocular video

- Frame/timestamp metadata and phase durations with event uncertainty.
- 2D normalized joint angles when the relevant joints are visible and motion is near the image plane.
- Relative body/ball positions normalized by person scale.
- Attempt-to-attempt consistency under matched view, distance and shot type.
- Image-plane landing/take-off alignment and release-to-apex timing.

### Class B: research estimates requiring stronger gates

- Camera-relative monocular 3D pose.
- Image-plane ball trajectory angle and velocity proxies.
- Hand/finger pose during partial ball occlusion.
- Camera-motion-compensated kinematics.

### Class C: unavailable from ordinary V1 input

- Metric release speed/height without calibration and scale.
- Ground-reaction force, joint torque, power and true kinetic-chain energy transfer.
- Ball spin without adequate high-speed imagery or direct tracking.
- Injury risk, diagnosis, fatigue state or causal explanation of a miss.

## Recommendation language contract

1. State the observed evidence and its frame/phase.
2. State comparison context: own history, matched exemplar, or research population.
3. State uncertainty/view dependency.
4. Offer one testable adjustment, not a causal guarantee.
5. Never convert a population mean into a pass/fail threshold without basketball-coach validation and independent outcome evidence.

Example:

> In this side-view free throw, estimated elbow extension at release is lower than in 4 of your 5 accepted free throws. The joint is visible, but hand occlusion makes the release frame uncertain by about two frames. Try one set keeping the same distance and camera while aiming for your previously repeatable release timing; this is a consistency cue, not proof that elbow angle caused the miss.
