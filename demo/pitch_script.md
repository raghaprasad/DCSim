# DCSim -- 5 Minute Video Pitch Script

Target: ~5 minutes. Sections are timed as rough guides.

---

## [0:00 - 0:40] The Problem

So you're on-call, and a customer calls in. Their training job is running slow -- or worse, it's stalled completely. They want to know why, and they want to know now.

Or maybe you're new to the team. Someone tells you "a single GPU thermal event can stall an entire 32-GPU training job." And you nod, but you don't really *see* it. How does one overheating GPU affect the other 31? Why does a 40-millisecond network blip add minutes to a training run?

These are hard things to explain with words. You kind of have to see it happen.

That's what DCSim does.

---

## [0:40 - 1:20] What is DCSim

DCSim is an interactive teaching tool for understanding GPU cluster failures. Think of it as a flight simulator -- but instead of flying a plane, you're watching a datacenter react to things going wrong.

You start with a healthy cluster. 32 GPUs, network switches, the cabling between them -- all modeled. Then you run a training job on it and inject failures. A GPU overheats. A network link goes down. A GPU throws a fatal error.

DCSim shows you exactly what happens next -- visually, step by step. Which GPUs stall. How long the job delays. Where the blast radius ends. No slides, no hand-waving. You see the failure cascade play out on an interactive timeline.

---

## [1:20 - 2:00] How It Works

Under the hood, we model a realistic cluster -- 4 nodes with 8 GPUs each, connected by high-speed NVLink within each node and InfiniBand across racks through Top-of-Rack and spine switches.

We run a training workload on this cluster. Each training iteration has two phases: a compute phase where every GPU crunches numbers, and a communicate phase where they share results. The important thing to know is that training is synchronous -- every GPU has to finish before anyone can move on. The slowest GPU sets the pace for the whole cluster.

Then we inject chaos -- thermal throttling, link failures, GPU crashes -- and the tool tracks the impact on every component, millisecond by millisecond.

One command generates an interactive HTML report you can explore in your browser.

---

## [2:00 - 3:40] Demo Scenarios

Let me walk you through four scenarios.

**Baseline -- this is what healthy looks like.** 32 GPUs, 10 training iterations. Each iteration takes 150 milliseconds -- 100ms of compute, 50ms of communication. Total training time: 1.5 seconds. Every GPU is green, every iteration is the same length. Nice and clean.

**Thermal throttle -- one hot GPU, everyone pays.** Same setup, but at 320 milliseconds, GPU 12 overheats. Its performance drops to a third of normal. Now here's the thing I was talking about -- training is synchronous. Look at the timeline. You can see 31 GPUs finish their compute and go gray -- idle, waiting. That one yellow GPU is still chugging along at 3x the normal time. Every other GPU is just sitting there, burning power, doing nothing. That's the visual you show a customer when they ask "why does one hot GPU matter?"

**Link flap -- the network hiccup nobody sees coming.** At 160 milliseconds, the uplink from rack 1 to the spine switch goes down. We're right in the middle of GPUs trying to share data. Everything stalls. The link comes back 40 milliseconds later, but now the network needs time to reroute -- another 10ms of overhead. Customers sometimes dismiss link flaps as "it was only down for 40 milliseconds." The timeline shows them exactly what those 40 milliseconds actually cost.

**XID 79 -- the one that triggers an escalation.** At 460 milliseconds, GPU 25 throws a fatal XID error and dies. The training job immediately stops. Now we wait -- 10 seconds for the system to detect the failure, reboot the GPU, and bring it back online. Look at the gap in the timeline. That's 10 seconds of all 32 GPUs doing absolutely nothing. When the GPU finally comes back, training picks up where it left off. Total time jumps from 1.5 seconds to over 11.5 seconds. One GPU, 8x the training time. That's the kind of thing on-calls need to be able to explain clearly during an incident.

---

## [3:40 - 4:20] Extensibility / What's Next

DCSim is built to be extended. The framework is modular -- hardware, workloads, chaos injection, and visualization are all separate layers. You can swap or add to any of them independently.

Want to model a different GPU? You can add H100 or A100 performance profiles. Different network topology? The system supports plugging in torus or dragonfly layouts. New failure modes like ECC memory errors or NVLink CRC failures? Add them as new chaos event types. New workloads like inference serving or pipeline parallelism? Implement the workload interface and it plugs right in.

Given more time, here's where I'd take this. First -- probabilistic failure models. Instead of manually injecting chaos events, define failure distributions based on real-world data. "GPUs fail at this rate, links flap with this frequency." Let the simulator generate realistic failure scenarios automatically. Second -- a multi-job scheduler, so you can see how failures affect not just one training job but a queue of jobs competing for GPUs. Third -- a live web dashboard instead of static HTML, so you can inject failures in real time and watch the cluster react. And longer term -- plug in real production telemetry to validate the simulator's predictions against what actually happens in the field.

---

## [4:20 - 4:50] Who This Is For

So who uses this?

If you're on-call, this helps you understand blast radius. Before a failure happens for real, you've already seen what a switch failure does to a rack, or how a GPU crash ripples through a training job.

If you're in product or sales, this is how you show a customer what resilience looks like -- and what happens when they don't have it. Way more convincing than a slide.

And if you're new to the team, this is a hands-on training tool. Run the scenarios, look at the timelines, build intuition. Ten minutes with DCSim teaches you more about failure propagation than a week of reading docs.

---

## [4:50 - 5:00] Closing

DCSim -- see what datacenter failures actually look like before they happen. One command, four scenarios, and a clear picture of what goes wrong and why.

Thanks for watching.
