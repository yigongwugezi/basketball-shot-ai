# P-M9 visible release measurement smoke

Primary product smoke: IMG_7216.MOV

- Upload API: HTTP 200
- Pose release frame: 135
- Ball observations: 19 / 19 requested frames
- TrustedFlight: frames 136-150, 15 unique observations, 0.467 s
- ReleaseEpoch interval: 4.500-4.533 s; representative 4.517 s
- Qualification: UNQUALIFIED
- Speed output: withheld; 95% profile interval 4.84-9.98 m/s
- Elevation output: withheld; 95% profile interval 76.4-80.0 degrees

The upload -> detector observations -> TrustedFlight -> ReleaseEpoch ->
ReleaseState chain completed. The V2 gate correctly prevented the prototype
from displaying point estimates because speed stability exceeded the frozen
availability threshold.

Additional integration observations:

- IMG_7215.MOV: HTTP 200, 12 observations / 0.367 s, UNQUALIFIED.
- BILI_003_A_BV1d84y1G7zq.mp4: HTTP 200, no qualifying unique-observation
  TrustedFlight segment.

These are product smoke observations, not a detector or scientific benchmark.
