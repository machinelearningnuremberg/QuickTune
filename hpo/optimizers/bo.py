
def BO(optimizer, metadataset, budget_limit, scale_curve=True,
                                                limit_by_cost=True,
                                                observe_cost=False):

    evaluated_configs = dict()
    optimizer_performance = [0]
    optimizer_budget = [0]
    optimizer_cost = [0]
    current_budget = optimizer_cost[-1] if limit_by_cost else optimizer_budget[-1]

    while current_budget < budget_limit:

        hp_index, budget = optimizer.suggest()
        print(f"Suggested conf: {hp_index}, budget: {budget}")
        cost = metadataset.get_curve_cost(hp_index, budget)

        if budget >= metadataset.get_curve_len(hp_index)-1:
            optimizer.converged_configs.append(hp_index)

        if len(optimizer.converged_configs) == len(optimizer.hp_candidates):
            print("All configs converged")
            break

        cost_curve_eval = metadataset.get_curve(hp_index, budget, curve_name="eval_time")
        cost_curve_train = metadataset.get_curve(hp_index, budget, curve_name="train_time")
        cost_curve = [x+y for x,y in zip(cost_curve_eval, cost_curve_train)]

        #cost_curve = np.cumsum(cost_curve).tolist()
        performance_curve = metadataset.get_curve(hp_index, budget)

        if scale_curve:
            performance_curve = [x/100 for x in performance_curve]

        if observe_cost:
            #this might break when we observe more than one epoch per step
            observed_cost = cost_curve[-1]
        else:
            observed_cost = None
        overhead_time = optimizer.observe(hp_index, budget, performance_curve, observed_cost)

        if hp_index in evaluated_configs:
            previous_state = evaluated_configs[hp_index]
            budget_increment = budget - previous_state[0]
            evaluated_configs[hp_index] = (budget, cost)
        else:
            budget_increment = budget
            evaluated_configs[hp_index] = (budget, cost)

        optimizer_budget.extend([i for i in range(optimizer_budget[-1]+1, optimizer_budget[-1]+budget_increment+1)])
        temp_cost = cost_curve[budget-budget_increment:budget]
        temp_cost = [x+overhead_time for x in temp_cost]
        optimizer_cost.extend(temp_cost)
        optimizer_performance.extend(performance_curve[budget-budget_increment:budget])
        current_budget = sum(optimizer_cost) if limit_by_cost else optimizer_budget[-1]


    for i in range(1,len(optimizer_performance)):
        max_perf = max(optimizer_performance[i-1:i+1])
        optimizer_performance[i] = max_perf
        optimizer_cost[i] += optimizer_cost[i-1]

    return optimizer_budget, optimizer_cost, optimizer_performance
