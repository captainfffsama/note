---
title: "Human-In-the-Loop Data Collection"
source: "https://aurelarnold.xyz/blog/human-in-the-loop-data-collection/"
author:
published: 2026-06-21
created: 2026-10-10
description: "Testing human-in-the-loop data collection on a real robot: when to step in, how to intervene, and whether it beats just collecting more demonstrations."
tags:
  - "clippings"
---
Human-In-the-Loop (HIL) data collection promises to improve a policy by demonstrating how to act in the situations where it struggles. Instead of collecting more data everywhere, you spend your effort exactly where the policy is weak, which should be a time-efficient way to reach a high success rate. I wanted to test this out for myself during my time at [Dream Machines](https://dream-machines.eu/), and work out how to get the best results out of it.

## Questions

When I first read about HIL data collection in the [π\*0.6](https://arxiv.org/abs/2511.14759) paper, the concept seemed simple and made a lot of sense: you improve a policy by collecting more data in the situations where it struggles. Easy! But once I wanted to do it myself, I realised it needed a bit more thought. It wasn’t quite that simple.

The data is usually collected by rolling out a policy while a human watches, stepping in whenever it struggles to demonstrate how to act in that situation. This data then trains the next iteration of the policy, which ideally performs better than the last. A prerequisite is a setup that lets you smoothly take over control from a running policy, so I implemented VR teleoperation for our setup at Dream Machines (more on that in my previous blog post: [VR teleoperation stack](https://aurelarnold.xyz/blog/vr-teleoperation-stack/)). Even with that in place, I had several open questions about how to get the best results:

- When exactly should I stop the policy?
- How should I do the interventions?
- How many interventions should I do before training a new policy iteration?
- Do I fine-tune the previous policy with the new data, or train a new policy from scratch on the aggregated dataset?

To find answers, I started by reading about the topic.

## Background

The idea of iteratively aggregating data and improving policies goes back to 2010, when Ross et al. introduced [DAgger](https://arxiv.org/abs/1011.0686) (Dataset Aggregation). Their goal is to imitate the behaviour of an expert, which is simply whatever provides the correct action, be it a policy, an algorithm, or a human. Instead of training the policy only on the states the expert visits during demonstrations, DAgger also trains it on the states the still-imperfect policy visits itself. The policy is rolled out to decide where to go and the expert is queried for the correct action at each visited state. The experts actions in the states explored by the policy is aggregated with the original dataset to train the next iteration. The policy’s own actions are only ever used to act and explore the state space, never as training labels.

[HG-DAgger](https://arxiv.org/abs/1810.02890) (Human-Gated DAgger) uses a human as the expert, which decides when to intervene. Instead of querying the exper at every state, a person watches the rollout and decides when to take over to demonstrate the right behaviour. Only the interventions are then aggregated with the initial demonstrations, and each new policy is retrained from scratch on this data (no fine-tuning).

[HIL-SERL](https://arxiv.org/abs/2410.21845) collects data the same way as HG-DAgger but uses it differently: instead of plain behaviour cloning, it optimises the policy with off-policy reinforcement learning, splitting the intervention data between a demonstration buffer and an RL buffer that are weighted differently during training. As the policy reaches higher success rates and faster cycle times, the human gradually intervenes less.

While the previous paper don’t specify how the interventions are done exactly, [RaC](https://arxiv.org/abs/2509.07953) (Recovery and Correction) introduces two rules about that:

1. When the policy makes a mistake, first recover into in-distribution states, then correct and finish the subtask.
2. End the rollout right after the intervention, so the data never contains states produced by a mix of policy and human control.

They show this is more data-efficient than HG-DAgger: the policy learns to recover and try again, which raises the success rate (at the cost of a longer average time on the successful episodes).

[π\*0.6](https://arxiv.org/abs/2511.14759) aggregates its data in an improvement loop of autonomous rollouts where expert operators step in to correct catastrophic failures and recover from mistakes. They don’t spell out the details (no explicit recover-then-correct rule, and no mention of when an episode ends), and they fine-tune each iteration from the pre-trained checkpoint rather than from the previous one.

Finally, the [LeRobot docs](https://huggingface.co/docs/lerobot/hil_data_collection) have a practical section on HIL data collection. They advise intervening when the robot is about to make an irreversible mistake, hesitates, or deviates from the expected trajectory. They recommend RaC’s first rule (recover, then correct) but not the second (they don’t terminate after an intervention but rather give control back to the policy and continue). They also fine-tune the policies iteratively rather than retraining from scratch. But their code examples are only for $\pi_{0}$, a much bigger policy for which retraining from scratch every iteration wouldn’t make sense anyway.

After reading all this, I settled on the following for my own experiments:

- Use LeRobot’s guidance on when to intervene.
- Recover into in-distribution states before correcting (RaC’s first rule).
- For small policies like mine, retrain from scratch on the aggregated dataset.
- Continue the episode after an intervention rather than terminating after the first one (so, not RaC’s second rule).

## Experiment

With these decisions made, I planned a set of experiments to see whether I could actually get useful results out of HIL data collection. I chose to train ACT policies, since they are small enough to train locally and I already had a working configuration and some experience with them (more on that in my previous blog post: [First Week of Imitation Learning](https://aurelarnold.xyz/blog/first-week-of-imitation-learning/)). I started by finding a good training configuration for the task, which I would then reuse throughout. The main question was whether collecting intervention data beats simply collecting more demonstrations of the full task. While running that comparison, though, I realised I still wasn’t sure how exactly the interventions should be done, so I added a third experiment comparing different intervention styles. The three experiments ended up being:

- Experiment 1: find a good ACT training configuration for the task.
- Experiment 2: compare collecting intervention data against collecting more demonstrations.
- Experiment 3: compare different intervention styles.
<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/task_demo.mp4" controls=""></video>

Figure 1. A demonstration of the cube pick-and-place task: pick the cube up on the left, hand it over to the right arm, and insert it into the slot.

I used a small cube pick-and-place task with three increasingly difficult subtasks (figure ). I kept the task rather simple and short, such that I can iterate more quickly. For evaluation I scored each episode by its task progress, recording the last subtask it finished successfully. This also lets me recover the overall success rate.

Progress score:

- 0.3: Pick up a cube on the left half of the table.
- 0.6: Hand the cube over to the right hand.
- 1.0: Place the cube into the square slot.

Every policy evaluation in the three experiments are done with 32 episodes. More details about the task and the evaluation are in the [Appendix](#appendix).

## Experiment 1: ACT Configuration

**Goal:** Find a good ACT training configuration for the task.

I took most of the parameters from my earlier ACT experiments ([First Week of Imitation Learning](https://aurelarnold.xyz/blog/first-week-of-imitation-learning/)), and for the batch size I used the largest that fit on our workstation. The one parameter really worth tuning for this task was the number of training steps, so I trained a single policy out to 320k steps and evaluated its checkpoints along the way.

### Procedure

1. Collect 100 demonstrations, creating the dataset $D_0$.
2. Train an ACT policy on $D_0$ up to 320k steps, storing intermediate checkpoints.
3. Evaluate the checkpoints \[6k, 10k, 20k, 40k, 60k, 80k, 120k, 160k, 320k\].

### Result

![Success rate of the base policy against training steps, with a 95% Wilson confidence band, peaking at 47%.](https://aurelarnold.xyz/assets/charts/hil-learning-curve.svg)

Figure 2. Success rate of the base policy against training steps, with the 95% Wilson confidence band.

The success rate jumps sharply between 20k and 40k steps (0% → 41%) and then plateaus, peaking at 47% at both 80k and 320k steps (figure ).

The policy also gets qualitatively smoother with more training, with cleaner transitions between action chunks. This is easiest to see at the early checkpoints (figure ).

<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/checkpoint_6k.mp4" controls=""></video>

Figure 3. The ACT policy at the 6k checkpoint (left) and the 80k checkpoint (right).

### Take-Away

Even at 320k steps, which is more than 120 epochs over the dataset, the performance doesn’t degrade. I suspect the task is simple enough that the policy just doesn’t overfit. Since the 80k and 320k checkpoints are almost indistinguishable (if anything, 320k is a touch smoother), I went with 80k steps for all the following experiments. That takes about 6 hours to train on our local GeForce RTX 5090. The full training configuration is in the [Appendix](#appendix).

## Experiment 2: Intervention vs Demonstration

**Goal:** Compare a policy trained on HIL intervention data against one trained on more demonstrations.

To make this comparison, I ran three rounds of data collection on top of the initial 100-episode dataset from Experiment 1. In each round I either collected 50 more full demonstrations, or rolled out the current policy for 50 episodes and intervened where needed, giving me two parallel branches of policies to compare.

For this experiment I corrected the policy after it made an error, that is, after it failed to place the cube into the slot. I recovered from the failure state (picked the cube back up), returned to an in-distribution state (holding the cube above the slot), and finished the subtask it had failed (inserted the cube). Figure shows one such intervention.

<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/correction_intervention.mp4" controls=""></video>

Figure 4. A correction intervention: after the policy fails to insert the cube, I take over, pick it up again, return to an in-distribution state above the slot, and insert it.

### Procedure

The steps in parentheses were already done in Experiment 1, so I carried their results over instead of redoing them. A superscript marks which branch a dataset or policy belongs to: I for the intervention branch and M for the branch that keeps adding more demonstrations.

1. (Collect 100 demonstrations, creating the dataset $D_0$.)
2. (Train the base policy $\pi_0$ on $D_0$ for 80k steps.)
3. For $i$ in \[1, 2, 3\], starting from $D^\text{M}_0 = D^\text{I}_0 = D_0$ and $\pi^\text{I}_0 = \pi_0$:
	1. More demonstrations: collect 50 new demonstrations and add them to $D^\text{M}_{i-1}$, giving $D^\text{M}_i$.
		2. Interventions: record 50 rollouts of $\pi^\text{I}_{i-1}$ with interventions and add the intervention segments to $D^\text{I}_{i-1}$, giving $D^\text{I}_i$.
		3. Train policy $\pi^\text{M}_i$ on $D^\text{M}_i$ and $\pi^\text{I}_i$ on $D^\text{I}_i$, each for 80k steps.
4. Evaluate all trained policies.

### Result

![Outcome composition over three collection rounds, comparing more demonstrations with interventions.](https://aurelarnold.xyz/assets/charts/hil-scaling.svg)

Figure 5. Outcome composition over the three collection rounds, comparing more demonstrations (left) with interventions (right). Each

The policies trained on intervention data improve faster and reach higher success rates than those trained on more demonstrations alone (figure ). The best success rate, 72%, comes after two rounds of intervention data. After a third round the policy starts to revert its own successful insertions, grabbing the cube again and then failing to re-insert it. These correction attempts mostly time out, running past the 60-second cap. The demonstration-only branch tops out at 38% after 200 episodes, and adding another 50 demonstrations doesn’t change that.

The interventions also got rarer as the policy improved. The share of rollouts that needed me to step in fell from 78% to 38% over the three rounds.

| round | rollouts | episodes w/ intervention |
| --- | --- | --- |
| 1 | 50 | 39 |
| 2 | 50 | 29 |
| 3 | 50 | 19 |

Because only the segments where I was in control are added, the intervention dataset grows slowly. After three rounds it holds 187 episodes and 123,930 frames, against 250 episodes and 203,834 frames for the demonstration dataset, even though it produced the better policy.

| round | demo episodes | demo frames | intervention episodes | intervention frames |
| --- | --- | --- | --- | --- |
| 0 | 100 | 80,831 | 100 | 80,831 |
| 1 | 150 | 119,285 | 139 | 100,151 |
| 2 | 200 | 162,498 | 168 | 113,930 |
| 3 | 250 | 203,834 | 187 | 123,930 |

Figure shows four behaviours I saw at the third round: a clean first-try success, a successful recovery and correction, a reverted success, and a timeout.

<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/round3_first_try.mp4" controls=""></video>

Figure 6. Behaviours of the round-3 intervention policy. Top-left: a first-try success. Top-right: a recovery and correction. Bottom-left: a reverted success, where the policy undoes a correct insertion. Bottom-right: a timeout.

### Take-Away

Two things stand out. First, the comparison is clear: the intervention data used less actual data than the extra demonstrations, yet it raised the success rate more. Second, the interventions changed the policy’s behaviour, not just its score. After one round it barely attempted any corrections. After two rounds it tried to correct essentially every failure, though it didn’t always succeed.

The way I ran the interventions also produced the odd round-three result where the policy reverts its own successes. It never learned to reliably tell whether the cube was already seated, so it would re-grab a correctly placed cube and then fail the re-insertion. Had I stopped each episode the moment the cube was correctly inserted, the success rate after three rounds would even have been 75%.

It is also worth noting that the gains didn’t come only from corrections. First-try successes went up as well, from 13% for the base policy to 59% after two rounds of interventions. That is probably down to how I did the interventions: after recovering the cube I always moved back above the slot and demonstrated the insertion from there, which adds more data for the very first insertion attempt too.

One process note. At first I evaluated each policy right after it finished training, which meant the evaluations happened on different days, at different times, under different lighting. It turned out those conditions can change a policy’s performance quite a lot, so I redid all the evaluations back-to-back on a single day. This is also why the number of interventions per round in the table above deviates from the results during the evaluation. The numbers above are from that second round, and the first-round results are in the [Appendix](#appendix) (figure ). In that first round I saw only a single correction attempt from $\pi^\text{I}_1$, and the over-correction, where the policy reverts its own successes, already showed up with $\pi^\text{I}_2$. Together these made me question whether I was intervening the right way, which is what motivated the third experiment.

## Experiment 3: Intervention Styles

**Goal:** Compare different intervention styles.

I compared four intervention styles:

- `nudge`: stop the policy just before it fails and correct with a small nudge.
- `redo`: stop the policy just before it fails, rewind, and redo the hard part.
- `recovery`: let the policy fail, then recover, return to an in-distribution state, and correct.
- `recovery+`: let the policy fail, rewind a little while recording, then take over to recover, return to an in-distribution state, and correct.

The motivation for `recovery+` came from that single correction attempt I saw after one round of `recovery` -style data. I suspected the problem was that taking over right after a failure drops the policy straight into states it has never seen. So with `recovery+` I first rewind the last few actions, walking the arm back through states it actually normally passed through, and only then take over. Crucially, the rewind is recorded too, so the dataset contains a path from the failure back through familiar states, when the cube isn’t placed correct.

`nudge` and `redo` come from the opposite worry. Once the policy started reverting its own successes, I wanted to see how far I could get with interventions that only improve the first-try success and never teach recovery at all.

For each style I collected a single round of intervention data over 50 episodes. Figure shows an example of each.

<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/nudge.mp4" controls=""></video>

Figure 7. The four intervention styles. Top-left: nudge. Top-right: redo. Bottom-left: recovery. Bottom-right: recovery+.

### Procedure

1. (Collect 100 demonstrations, creating the dataset $D_0$.)
2. (Train the base policy $\pi_0$ on $D_0$ for 80k steps.)
3. For each style $n$ in \[nudge, redo, recovery, recovery+\]:
	1. Record 50 rollouts of $\pi_0$ with $n$ -style interventions and add them to $D_0$, giving $D^\text{n}_1$.
		2. Train $\pi^\text{n}_1$ on $D^\text{n}_1$ for 80k steps.
4. Evaluate all trained policies.

### Result

How much I had to intervene during collection, and how much data that added, varied by style. The intervention depth is the median fraction of an episode I spent in control, so a shallow depth means short interventions:

| style | rollouts intervened | intervention depth (median %) | intervention frames |
| --- | --- | --- | --- |
| nudge | 31/50 (62%) | 8% | 2,100 |
| redo | 35/50 (70%) | 34% | 9,049 |
| recovery | 39/50 (78%) | 39% | 19,320 |
| recovery+ | 26/50 (52%) | 40% | 14,877 |

![Outcome composition after one intervention round for each of the four styles, against the base policy. A capped black error bar on each bar marks the success rate and its 95% confidence interval.](https://aurelarnold.xyz/assets/charts/hil-styles.svg)

Figure 8. Outcome composition after one intervention round for each style, against the base policy. The black error bar on each bar marks the 95% Wilson confidence interval of the success rate (counting successes with and without correction).

After a single round, `recovery+` reaches the highest success rate (56%), with a lot of first-try successes (figure ). `recovery` is close behind (53%) and makes many correction attempts, succeeding on roughly half of them. `nudge` and `redo` both improve on the base policy (25%), with `redo` (53%) ahead of `nudge` (44%).

### Discussion

`nudge` beats the base policy but trails the other three styles. I think the reason is simply how little data a nudge adds: the corrections are so short that there isn’t much signal to learn a new skill, and it would probably need many more episodes. And since a fixed number of intervention episodes takes about the same time regardless of style, `nudge` also collects the least data per unit of time by a wide margin.

`recovery+` gives the best results here, but it doesn’t really lend itself to more than one round. In the sections where I recorded the rewind to teach the correction, the resulting policy ($\pi^\text{recovery+}_1$) sometimes hesitates, flickering between continuing and reversing (figure ). Through the recorded rewinds, that hesitant behaviour would feed straight into the next round’s dataset, which isn’t what I want.  
`recovery+` 在这里效果最佳，但它并不适合多轮训练。在我录制回放过程以训练纠正策略的部分，得到的策略（ $\pi^\text{recovery+}_1$ ）有时会犹豫，在继续和回放之间摇摆不定（图 9）。通过录制的回放过程，这种犹豫行为会直接传递到下一轮的数据集中，这不是我想要的结果。

<video src="https://aurelarnold.xyz/assets/videos/human-in-the-loop-data-collection/recovery_plus_hesitation.mp4" controls=""></video>

Figure 9. The recovery+ policy hesitating near the slot, switching between continuing and reversing. 图 9. recovery+ 策略在槽附近犹豫不决，在继续和反转之间切换。

Comparing `recovery` and `recovery+` on the episodes where the first insertion failed, `recovery` actually attempts a correction in more of them (in 31% of all evaluation episodes, versus 19% for `recovery+`). Since the two styles differ only in how the correction is started, this suggests the explicit rewinding isn’t necessary: simply letting the operator “rewind” by physically moving the arm back after a failure is already enough to elicit the correction attempts.  
在首次插入失败的场景中，比较 `recovery` 和 `recovery+` ， `recovery` 实际上在更多场景中尝试进行纠错（在所有评估场景中占 31%，而 `recovery+` 仅占 19%）。由于这两种方法的区别仅在于纠错的启动方式，这表明显式的回退并非必要：只需让操作员在失败后通过物理移动机械臂来“回退”即可触发纠错尝试。

`redo` improves on the base purely by raising the first-try success rate, since it never teaches the policy to recover. Interestingly, `recovery+` improved the first-try rate about as much as `redo` did. The same effect I described in Experiment 2, where always lifting the cube back above the slot before re-inserting adds data for the first approach too.  
`redo` 仅通过提高首次尝试成功率来改进基础策略，因为它从未教会策略如何恢复。有趣的是， `recovery+` 对首次尝试成功率的提升与 `redo` 几乎相同。这与我在实验 2 中描述的效果相同，即在重新插入之前始终将立方体抬回到槽位上方，也会为第一次尝试增加数据。

One thing I would change: I fixed the number of rollouts (50) per style rather than the number of interventions, so the styles don’t all rest on the same amount of intervention episodes (see the table above).  
有一点需要改进：我固定了每种风格的推广次数（50），而不是干预次数，这样不同的风格就不会都依赖于相同数量的干预事件（见上表）。

While watching $\pi^\text{recovery+}_1$ and seeing how often it corrected, I went back to re-evaluate $\pi^\text{recovery}_1$, which I had first evaluated a few days earlier (during Experiment 2’s first evaluation round) and where I had seen only a single correction. The re-evaluation looked completely different, with lots of corrections. The only visible difference between the two recordings is the light in the window. The first evaluation looks like it was recorded early in the morning, while the re-evaluation had bright sunlight coming through (figure ).  
在观察 $\pi^\text{recovery+}_1$ 及其纠正频率的同时，我重新评估了 $\pi^\text{recovery}_1$ 。几天前（在实验 2 的第一轮评估中），我曾评估过 $\pi^\text{recovery}_1$ ，当时只观察到一次纠正。重新评估的结果截然不同，出现了大量的纠正。两次记录之间唯一明显的区别是窗户的光线。第一次评估看起来像是清晨录制的，而重新评估时阳光明媚（图 10）。

![The scene during the first evaluation of the recovery policy, in cooler morning light.](https://aurelarnold.xyz/assets/human-in-the-loop-data-collection/lighting_warm.jpg)

Figure 10. The same scene during the two evaluations of the recovery policy, a few days apart. Left: the first evaluation, in cooler morning light. Right: the second re-evaluation, in bright sunlight. 图 10. 同一场景在两次评估恢复策略期间的对比，两次评估相隔数日。左图：第一次评估，在较为凉爽的晨光下进行。右图：第二次复评，在明亮的阳光下进行。

Because of that, I also re-evaluated the base policy $\pi_0$ to keep the comparison fair, and ran every evaluation for this third experiment back-to-back on the same morning under consistent light (figure in the [Appendix](#appendix)).  
因此，为了保持比较的公平性，我还重新评估了基本策略 $\pi_0$ ，并在同一天早上在一致的光照条件下连续运行了第三个实验的每一次评估（附录中的图 14）。

## Conclusion 结论

Across these experiments, HIL data collection did what it promised for the small ACT policy I trained. Spending the same time budget on interventions instead of extra demonstrations gave me a better performing policy, and it got there faster: the success rate climbed to 72% after two rounds of interventions, while collecting more full demonstrations plateaued around 38%.  
在这些实验中，HIL 数据收集确实如预期那样，对我训练的小型 ACT 策略起到了作用。将相同的时间预算用于干预而非额外的演示，使我的策略性能更佳，而且见效更快：经过两轮干预后，成功率攀升至 72%，而收集更多完整演示的成功率则停滞在 38% 左右。

But the experiments also showed that the details of the intervention matter as much as the decision to intervene at all. The same `recover` procedure that pushed the success rate up also taught the policy to second-guess itself: by the third round it would undo its own correct insertions. I think this comes from how I recorded the corrections. I stopped right after re-inserting the cube, so each correction ends on the insertion and never on leaving the placed cube alone and returning home. The original demonstrations do show that ending, but the corrections kept adding data for re-grabbing. Comparing intervention styles, the ones that demonstrate recovery (`recovery`, `recovery+`) gave the best results. `recovery+` came out slightly ahead, but its recorded rewinds make it hard to iterate, and plain `recovery` got most of the way there.  
但实验也表明，干预的细节与是否进行干预同样重要。同样的 `recover` 程序虽然提高了成功率，但也让策略学会了自我怀疑：到了第三轮，它就会撤销自己正确的插入操作。我认为这源于我记录纠正操作的方式。我在重新插入立方体后立即停止记录，因此每次纠正操作都止于插入，而不是停留在放置立方体后返回初始位置。最初的演示确实显示了这种结束方式，但纠正操作会持续增加重新抓取的数据。比较不同的干预方式，能够实现恢复的干预方式（ `recovery` 、 `recovery+` ）效果最佳。 `recovery+` 略胜一筹，但由于其记录的回溯操作难以迭代，而普通的 `recovery` 也基本达到了预期效果。

So, to get the best results out of HIL data collection, here are my main take-aways:  
因此，为了从 HIL 数据采集中获得最佳结果，以下是我的主要心得：

- Recover into in-distribution states after a failure, then solve the failed part the same way as in the demonstrations. That teaches the policy to recover, and because the correction approaches the goal just like a demonstration does, it lifts the first-try success rate too.  
	失败后恢复到分布内状态，然后以与演示中相同的方式解决失败的部分。这教会了策略如何恢复，并且由于纠正过程与演示过程一样接近目标，因此也提高了首次尝试的成功率。
- Additionally I would finish the whole subtask when doing a correction, instead of stopping the moment the cube is seated, so the policy does not overfit to correcting even when it has already succeeded.  
	此外，我会在进行纠正时完成整个子任务，而不是在立方体就位后立即停止，这样即使已经成功，该策略也不会过度拟合以进行纠正。

About running the experiment itself, here is what I would do differently next time:  
关于实验本身的运行，下次我会做出以下不同的调整：

- Fix the number of interventions per round rather than the number of rollouts, so every round and style is compared on the same amount of intervention data (π\*0.6 fixes the interventions this way too).  
	固定每轮干预次数，而不是推广次数，这样每一轮和每种风格都可以在相同数量的干预数据上进行比较（π\*0.6 也可以这样固定干预次数）。
- Evaluate every policy back-to-back under the same conditions. This one wasn’t really about HIL at all, and it was the biggest surprise of the project: a change in daylight alone was enough to turn “one correction” into “lots of corrections” for the same policy.  
	在相同条件下对每项策略进行逐一评估。这其实与硬件在环（HIL）无关，但却是该项目最大的意外：仅仅是日光的变化就足以让同一项策略的“一次修正”变成“多次修正”。
- Choose a longer task with more subtasks. Even the base policy, trained on just the initial 100 demonstrations, cleared the first two subtasks every time, so the policies only ever differed on the final insertion. More steps that are not solved straight away would spread them across the progress score and make them easier to compare.  
	选择包含更多子任务的更长任务。即使是仅基于最初 100 个示例训练的基础策略，每次都能完成前两个子任务，因此策略之间的差异仅在于最后的插入步骤。更多无法立即解决的步骤会分散到进度分数中，从而更容易进行比较。

The questions I am most curious about now are about transfer. Would the same intervention recipe carry over to other tasks, and to other kinds of policies, especially the larger VLAs that are fine-tuned rather than retrained from scratch?.  
我现在最感兴趣的问题是关于迁移性。同样的干预方案能否推广到其他任务和其他类型的策略，特别是那些规模更大、需要微调而非从头开始重新训练的 VLA（虚拟逻辑分析系统）？

## Appendix 附录

### Task 任务

The task used for the experiments is a small cube pick-and-place task with three increasingly difficult subtasks. The task is to:  
实验中使用的任务是一个小型立方体拾取放置任务，包含三个难度递增的子任务。该任务的目标是：

1. pick up a cube on the left half of the table,  
	拿起桌子左侧的一个立方体，
2. hand it over to the right hand,  
	交给右手边的人，
3. place it into a square slot.  
	将其放入方形槽中。

Figure shows a video of one demonstration of the task.  
图 1 显示了该任务的一个演示视频。

The cube is 30×30×30 mm. It is placed at a random position inside a marked 10×10 cm square, always in the same orientation (figure , left). The slot is 31×31 mm and always in the same location (figure , right).  
立方体尺寸为 30×30×30 毫米。它被随机放置在一个标记好的 10×10 厘米正方形内，方向始终相同（图 11，左）。槽口尺寸为 31×31 毫米，位置始终相同（图 11，右）。

![The 10x10 cm region the cube is randomly placed in, with several start positions overlaid.](https://aurelarnold.xyz/assets/human-in-the-loop-data-collection/start_region.jpg)

Figure 11. Left: the 10×10 cm region the cube is randomly placed in, with several start positions overlaid. Right: the slot and the cube. 图 11. 左图：立方体随机放置在 10×10 厘米的区域内，图中叠加了几个初始位置。右图：槽口和立方体。

### Evaluation 评估

For evaluation I place the cube on an 8×8 grid of positions inside the 10×10 cm square, using every second grid cell, which gives 32 trials per policy (figure ).  
为了进行评估，我将立方体放置在 10×10 厘米正方形内的 8×8 网格位置上，使用每隔一个网格单元格，这样每个策略就有了 32 次试验（图 12）。

![The 8x8 grid of cube positions used for evaluation, marked inside the start square.](https://aurelarnold.xyz/assets/human-in-the-loop-data-collection/eval_grid.jpg)

Figure 12. The 8×8 grid of cube positions used for evaluation, giving 32 trials per policy. 图 12. 用于评估的 8×8 立方体位置网格，每个策略进行 32 次试验。

I rate each policy with the task-progress score: if it fails a subtask, it keeps the score of the last subtask it completed. The success rate then falls out of the scores, as the share of episodes that reach 1.0.  
我根据任务进度得分对每项策略进行评分：如果策略未能完成某个子任务，则保留其已完成的最后一个子任务的得分。成功率（即达到 1.0 分的策略所占的比例）则从总分中剔除。

Progress score:进步分数：

1. failure: 0.0 失败：0.0
2. pick\_up: 0.3 (the cube leaves the ground while in contact with both sides of the gripper).  
	pick\_up: 0.3（立方体在与夹爪两侧接触时离开地面）。
3. hand\_over: 0.6 (the right gripper holds the cube by itself).
4. place: 1.0 (the cube is fully inserted and correctly oriented in the slot, and the arms return to the rest pose).

On top of the score, the operator tags the failure mode during evaluation, which allows for a more fine-grained analysis afterwards.

I found it helpful to divide the task roughly into a small number of subtasks with a clear definition for each, so I don’t have to think about the score while evaluating. The failure-mode tags, on the other hand, are easiest to add live, since you only learn the exact failure mode by watching.

The operator stops the episode manually when the arms return to the rest pose, the policy gets stuck, or it makes an unrecoverable failure, and automatically after 60 seconds. A demonstration takes about 16 seconds on average, so the 60-second cap still leaves room for several recovery attempts.

### ACT training configuration

- `chunk_size`: 50
- `n_action_steps`: 50
- `kl_weight`: 10.0
- `optimizer_lr`: 3e-5
- `optimizer_lr_backbone`: 3e-5
- `batch_size`: 32
- `steps`: 80k

### Experiment 2: Evaluation 1

In Experiment 2’s first evaluation round, I evaluated each policy right after it trained, which meant different days and different lighting. The $\pi_0$ evaluation reuses the one from Experiment 1. Figure shows the intervention branch across those rounds in the first evaluation: self-correction grows, but by the last round (round 2) a sizeable share of episodes time out or revert their own success.

![Outcome composition across the first three intervention rounds in the first evaluation, where over-correction grows.](https://aurelarnold.xyz/assets/charts/hil-overcorrection.svg)

Figure 13. Outcome composition across the first three rounds of collection (0, 1, 2) in the first evaluation round.

### Experiment 3 re-evaluation

After finding how much the lighting moved the results, I ran every evaluation for the third experiment back-to-back on the same morning, under the same light. Figure shows the starting scene of episode 0 for each of the four style re-evaluations, which looks consistent across all of them.

![Episode 0 of the nudge re-evaluation.](https://aurelarnold.xyz/assets/human-in-the-loop-data-collection/ep0_nudge.jpg)

Figure 14. The starting scene of episode 0 for each re-evaluation, in order: nudge, redo, recovery, recovery+. The light is consistent across all four.