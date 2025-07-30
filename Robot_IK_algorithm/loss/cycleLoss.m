function loss = cycleLoss(predPose, targetPose, cfg)
% cycleLoss: FK-loss (保持 dlarray 计算图)
% 输入:
%   predPose   42×B dlarray
%   targetPose 42×B dlarray
%   cfg        配置，包含 w_pos, w_rot, beta_pos, beta_rot

    % 确保两者是 dlarray
    predPose   = dlarray(predPose);
    targetPose = dlarray(targetPose);

    % 位置 Smooth L1
    posLoss = smoothL1_dl(predPose(1:30,:), targetPose(1:30,:), cfg.beta_pos);

    % 旋转 Smooth L1
    rotLoss = smoothL1_dl(predPose(31:end,:), targetPose(31:end,:), cfg.beta_rot);

    % 合并损失 (保持 dlarray)
    loss = cfg.w_pos * posLoss + cfg.w_rot * rotLoss;
end

function l = smoothL1_dl(pred, target, beta)
% Smooth L1 (Huber) 损失, 完全可微
    diff = abs(pred - target);
    beta = cast(beta, underlyingType(pred));  % 匹配 pred 类型

    % 平滑 L1 损失 (无逻辑运算，纯 min)
    quadratic = 0.5 * (diff.^2) ./ beta;
    linear = diff - 0.5 * beta;

    % 使用 min 保持梯度连续
    l = mean(min(quadratic, linear), 'all');
end
