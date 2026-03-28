Here is the sort of demo I would want to give

  "Imagine a 32 GPU cluster that involved in a training run. They are scheduled to finish in 100ms and then share their data as part of every collective step. At 50ms our chaos monkey injects a XID72 error on GPU #8 causing it to not funciton anymore. Assuming that a reboot and reload takes 10's of minutes, the engine instantly updates GPU #8's completion event timeline pushing it out by several minutes (converted to appropriate milliseconds). The other GPUs finish and are blocked on GPU 8 and sim shows them idling waiting for the broken GPU.

  Likewise some of the choas events that I want to represent
  - GPU" Thermal throttling -- slowdown of GPU perf. this means if every collective step took 100ms then a thermal throttle 3x the time
  - GPU XID error - nonfunctional GPU. needs to reboot and repair before resuming (adds a standard block time for detection, reboot and rebring up of the GPU)
  - Link flap

  Here is another example for link flap "Imagine  32 GPUs finishes their compute phase they are healthy and want to exchange the gradients (comms phase). under perfect conditions this takes 20 ms -- now chaos monkey injects a link flap -- during time we need to recalcuate the routing this takes 10ms

  During this time rack1 is isolated and cannot send data to rack 2
  After recalculating the new route the comms phase resumes -- now the recalculation causes a traffic jam slows the iterations by several 100s of ms



## basline
- 32 GPUs 
- 10 cont training iterations 
- each iteration consists of 100ms compute phase and 50ms all-reduce comms / network phase 
-  baseline pref : one iteration takes 150ms 

## Test case 1: thermal throttle 
- 32 GPUs
- 10 training iterations 
- chaos event at t=320 (compute phase of 3rd iteration) -- GPU 12 is hit with thermal issues 
- logs should show that between 400 and 480 all 32 GPUS are idle waiting for GPU 12 to finish 


## Test case 2: link flap 
- 32 GPUs
- 10 training loop
- event at 160 -- rack1 (g0 to g7) uplink failure -- recovers only by t=200 
- logs show that staggerd completion because of degraded path - proving iteration took longer than the baseline target 




## Test Case 3: xid 79 
- 10 iteration loop
- t=460 G25 has xid error 
- logs indicate 10000 ms penalty to find new healthy GPU
