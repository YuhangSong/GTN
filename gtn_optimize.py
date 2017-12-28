if args.algo in ['a2c', 'acktr']:
    values, action_log_probs, dist_entropy, conv_list = actor_critic.evaluate_actions(Variable(rollouts.states[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))
    # pre-process
    values = values.view(args.num_steps, num_processes_total, 1)
    action_log_probs = action_log_probs.view(args.num_steps, num_processes_total, 1)

    # compute afs loss
    afs_per_m_temp, afs_loss = actor_critic.get_afs_per_m(
        action_log_probs=action_log_probs,
        conv_list=conv_list,
    )
    if len(afs_per_m_temp)>0:
        afs_per_m += [afs_per_m_temp]

    if (afs_loss is not None) and (afs_loss.data.cpu().numpy()[0]!=0.0):
        afs_loss.backward(mone, retain_graph=True)
        afs_loss_list += [afs_loss.data.cpu().numpy()[0]]

    advantages = Variable(rollouts.returns[:-1]) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    final_loss_basic = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef

    ewc_loss = None
    if j != 0:
        if ewc == 1:
            ewc_loss = actor_critic.get_ewc_loss(lam=ewc_lambda)
    
    if ewc_loss is None:
        final_loss = final_loss_basic
    else:
        final_loss = final_loss_basic + ewc_loss

    basic_loss_list += [final_loss_basic.data.cpu().numpy()[0]]
        
    final_loss.backward()

    if args.algo == 'a2c':
        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

    optimizer.step()

elif args.algo == 'ppo':
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    old_model.load_state_dict(actor_critic.state_dict())
    if hasattr(actor_critic, 'obs_filter'):
        old_model.obs_filter = actor_critic.obs_filter

    for _ in range(args.ppo_epoch):
        sampler = BatchSampler(SubsetRandomSampler(range(num_processes_total * args.num_steps)), args.batch_size * num_processes_total, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)
            if args.cuda:
                indices = indices.cuda()
            states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
            actions_batch = rollouts.actions.view(-1, action_shape)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy, conv_list = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

            _, old_action_log_probs, _, old_conv_list= old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
            adv_targ = Variable(advantages.view(-1, 1)[indices])
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            optimizer.zero_grad()
            (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
            optimizer.step()
