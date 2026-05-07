# Herd Experiment Results Interpretation

These notes interpret the 15-condition experiment using the final 50-generation mean values in `report_table.csv`. The most useful overall metric is the cluster-survival score, calculated as:

```text
survival rate x protected time fraction
```

This score rewards conditions where sheep both survive and spend more time in protective clusters. It does not directly reward clustering in the evolutionary fitness function.

## Fitness Functions Used

All sheep fitness treatments used the same general structure:

```text
if grass_found == 0:
    fitness = 0.001
else:
    forage_reward = grass_reward x grass_found
    speed_reward = quick_grass_reward x weighted_grass
    survival_reward = survival_bonus if sheep_survived else 0
    death_multiplier = death_penalty if sheep_was_eaten else 1

    fitness = max(
        0.001,
        (forage_reward + speed_reward + survival_reward) x death_multiplier
    )
```

The `weighted_grass` term gives more reward when grass is found sooner after the previous food item:

```text
weighted_grass += 1 / (1 + timer_since_last_grass / 60)
```

The three fitness treatments were:

```text
Current fitness:
fitness = max(0.001, (8 x grass_found + 6 x weighted_grass + 0) x death_multiplier)
death_multiplier = 0.2 if eaten, otherwise 1
```

```text
Survival bonus fitness:
fitness = max(0.001, (8 x grass_found + 6 x weighted_grass + 40_if_survived) x death_multiplier)
death_multiplier = 0.2 if eaten, otherwise 1
```

```text
Strong survival fitness:
fitness = max(0.001, (6 x grass_found + 4 x weighted_grass + 80_if_survived) x death_multiplier)
death_multiplier = 0.05 if eaten, otherwise 1
```

None of these fitness functions directly rewards cluster size, protected time, or being near other sheep. Clustering is measured as an outcome using the objective metrics in the result CSVs.

## Main Findings

- The strongest overall condition was **Test 5 with strong survival fitness**. It achieved the highest survival rate (84.0%), a high protected-time fraction (28.4%), the lowest wolf kill rate (3.84 kills), and the highest cluster-survival score (0.242).
- **Test 2 with survival bonus fitness** was the second-best condition. It had slightly lower survival than Test 5 strong survival (80.8% vs 84.0%) but the best foraging performance of all conditions (11.12 grass per sheep).
- The best conditions combine survival pressure with either realistic fear logic or cluster protection. This suggests that clustering improves when survival is made more important, even though cluster membership itself is not directly rewarded.
- Test 5 current fitness performed poorly compared with Test 5 strong survival. Survival increased from 34.9% to 84.0%, protected time increased from 9.8% to 28.4%, and wolf kills fell from 15.62 to 3.84 when strong survival pressure was used.
- Test 4 strong survival produced the highest protected-time fraction (29.0%), which supports the idea that fear triggered by being unprotected can encourage safety-seeking behaviour. However, it did not outperform Test 5 strong survival overall because survival and foraging were both lower.
- Test 3 performed weakly under the current fitness setup. The combination of wolf fear and cluster protection produced low survival (10.1%) and low protected time (5.6%), suggesting that fear of wolves alone was not enough for sheep to reliably exploit safety in numbers under the original fitness function.
- The survival bonus treatment was not universally beneficial. It helped Test 2 strongly, but made Test 5 worse than both current and strong survival fitness. This suggests that simply adding a survival reward is not enough; the balance between foraging reward, survival reward, and predation penalty matters.

## Interpretation by Graph

### Survival Rate

- The survival graph shows that **Test 5 strong survival** is the clearest survival winner, reaching an 84.0% final-window survival rate.
- **Test 2 survival bonus** also performs strongly at 80.8%, indicating that cluster protection alone can be very effective when survival is rewarded.
- The poorest survival conditions are Test 3 current (10.1%), Test 5 survival bonus (16.8%), Test 2 strong survival (20.3%), and Test 3 survival bonus (21.5%).
- A key point for the report: survival pressure can substantially improve survival, but its effect depends strongly on the behavioural rule being tested.

### Protected Time Fraction

- Protected time measures how much sheep-time was spent in a cluster large enough to count as protected.
- **Test 4 strong survival** had the highest protected-time fraction (29.0%), slightly above Test 5 strong survival (28.4%), Test 2 survival bonus (27.6%), and Test 1 current (27.5%).
- This supports the hypothesis that sheep can show stronger clustering when fear is linked to being unprotected.
- However, protected time alone is not sufficient. Test 2 strong survival had relatively high protected time (22.8%) but very poor survival (20.3%) and poor foraging (3.45 grass per sheep).

### Cluster-Survival Objective

- The cluster-survival objective is the clearest single summary metric because it combines survival with clustering behaviour.
- The top four conditions were:
  - Test 5 strong survival: 0.242
  - Test 2 survival bonus: 0.226
  - Test 1 current: 0.202
  - Test 4 strong survival: 0.199
- Test 5 strong survival is the best all-round result because it combines high survival, high protected time, good foraging, and low wolf kills.
- Test 4 strong survival is important biologically because it produced the highest protected-time fraction, but its lower foraging and survival made it rank fourth overall.

### Foraging Performance

- The highest grass-per-sheep value was **Test 2 survival bonus** at 11.12 grass per sheep.
- Test 5 strong survival was close behind at 10.73 grass per sheep while also achieving much better survival and lower wolf kills than most conditions.
- Test 2 strong survival performed badly on foraging (3.45 grass per sheep), suggesting that excessive survival pressure can reduce foraging efficiency in some behavioural setups.
- This matters because clustering is only useful if it does not destroy foraging performance. Test 5 strong survival is therefore more promising than conditions that cluster but forage poorly.

### Wolf Kills

- Lower wolf kills indicate better anti-predator performance.
- Test 5 strong survival had the lowest wolf kills (3.84), followed by Test 2 survival bonus (4.60).
- Test 3 current had the highest wolf kills (21.58), followed by Test 5 survival bonus (19.98) and Test 2 strong survival (19.12).
- The wolf-kill results support the survival-rate results: the best conditions are not just inflating sheep fitness, they are actually reducing predation.

## Fitness Function Interpretation

- The **current fitness function** can produce reasonable results in some cases, especially Test 1, but it does not consistently promote survival or clustering.
- The **survival bonus** fitness can work well when cluster protection is available without fear input, as seen in Test 2. However, it can perform badly in Test 5, so it is not a general solution.
- The **strong survival** fitness is the most promising fitness treatment overall. It produced the best condition in Test 5 and improved Test 4 substantially.
- The results suggest that improving clustering behaviour indirectly is possible by increasing selection pressure against predation, rather than rewarding clustering directly.
- Because clustering was measured as an objective outcome rather than added as a reward, the strongest results are less vulnerable to the criticism that sheep were merely optimising an explicit cluster reward.

## Biological Interpretation

- The best-performing condition, Test 5 strong survival, is also the most biologically plausible fear rule: sheep become afraid when a wolf is seen and they are not protected.
- This condition suggests that sheep can benefit from safety in numbers when fear is conditional on both predator presence and lack of protection.
- Test 4 strong survival supports the safety-seeking hypothesis: when being unprotected triggers fear, sheep spend the most time protected.
- Test 2 survival bonus shows that cluster immunity alone can create a survival advantage, but it does not demonstrate learned safety-seeking in the same way as Test 4 or Test 5.
- Test 3 suggests that wolf fear plus cluster immunity is not necessarily enough for sheep to infer safety in numbers unless the fitness pressure strongly favours survival.

## Report-Ready Conclusion

- Overall, the results support the hypothesis that clustering can emerge as an anti-predator strategy when survival pressure is sufficiently strong.
- The best-performing condition was Test 5 with strong survival fitness, which combined the most realistic fear rule with strong selection against predation.
- This condition achieved the best balance of survival, clustering, foraging, and predator avoidance.
- The results do not support directly rewarding clustering as necessary. Clustering improved under survival-based fitness changes, meaning the behaviour can be evaluated as an emergent consequence of survival pressure.
- The strongest evidence for improved clustering behaviour is the increase in protected time and cluster-survival score under strong survival pressure, especially in Test 5 and Test 4.

## Caveats

- These results use a fixed random seed. This is useful for controlled comparison, but the findings should eventually be checked across multiple seeds.
- The experiment compares final 50-generation means, which smooths late-training performance but may hide instability during earlier generations.
- Mean end cluster size is calculated only when at least one sheep survives, so it should be interpreted alongside survival rate.
- The cluster-survival score is an objective summary metric, not a biological law. It is useful for ranking conditions, but individual metrics should still be reported separately.
- Strong survival fitness changes the evolutionary pressure substantially, so improvements should be interpreted as evidence that survival pressure can promote clustering, not as proof that the fear rule alone is sufficient.
