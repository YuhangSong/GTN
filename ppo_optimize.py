if args.algo in ['a2c', 'acktr']:
    values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(rollouts.states[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))
   
    values = values.view(args.num_steps, args.num_processes, 1)
    action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

    advantages = Variable(rollouts.returns[:-1]) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()
    (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

    if args.algo == 'a2c':
        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

    optimizer.step()
elif args.algo == 'ppo':
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    '''copy all model parameter to old_model'''
    old_model.load_state_dict(actor_critic.state_dict())

    for _ in range(args.ppo_epoch):
        sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps)), args.batch_size * args.num_processes, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)
            if args.cuda:
                indices = indices.cuda()
            states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
            actions_batch = rollouts.actions.view(-1, action_shape)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

            _, old_action_log_probs, _ = old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
            adv_targ = Variable(advantages.view(-1, 1)[indices])
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            optimizer.zero_grad()
            (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
            optimizer.step()
