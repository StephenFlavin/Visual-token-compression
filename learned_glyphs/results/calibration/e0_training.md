# E0 training calibration

| channel | em | h | px2/char | CER | CER via other eval channels | Phase-2 latin CER | steps |
|---|---|---|---|---|---|---|---|
| default | 5 | 6 | 15.32 | 0.0098 | clean=0.0098 harsh+heldout=0.8142 | 0.005153145695364238 | 3500 |
| default | 6 | 7 | 21.60 | 0.0023 | clean=0.0023 harsh+heldout=0.7947 | 0.002918046357615894 | 3500 |

P@3% (px2/char at CER 3%, isotonic + log-linear interpolation, vtc.analysis.p_at):

* Phase-2 latin (phase2_main): 15.110262199297157
* train default / eval clean: 15.319151466342507 (1.01x Phase 2)
  * eval through clean: 15.319151466342507 (1.01x Phase 2)
  * eval through harsh+heldout: None

Calibration target (PLAN E0): a preset within ~15% of Phase-2 latin P@3%.
