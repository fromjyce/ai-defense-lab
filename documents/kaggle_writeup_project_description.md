### Overview

Closed-loop red-team / blue-team payment fraud system: a synthetic-transaction
generator feeds a LightGBM fraud detector, an evolutionary attacker mutates
transactions against the detector's own score until it evades, successful
evasions are mined back into training, and the detector retrains. We track
attack success and clean-set PR-AUC together every generation so a detector
can't "win" by turning into a blanket blocker.

##### JNR

- Rohith R (Lead) - roahith11@gmail.com
- Jayashre - jaya2004kra@gmail.com
- Nidhi Gummaraju - nidhigumm05@gmail.com
