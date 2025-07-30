function loss = smoothL1Loss(pred, target, beta)
% smoothL1Loss 带 beta 参数的 Huber 损失（PyTorch 风格）
% 输入:
%   pred   - 预测值向量
%   target - 目标值向量
%   beta   - 平滑参数（阈值）
% 输出:
%   loss   - 标量损失值

    % diff = abs(pred - target);
    % loss = sum( ...
    %     (diff < beta) .* (0.5 .* diff.^2 ./ beta) + ...
    %     (diff >= beta) .* (diff - 0.5 * beta), 'all');
    diff = pred - target;
        abs_diff = abs(diff);
        loss = sum( ...
            (abs_diff < beta) .* 0.5 .* (diff.^2) / beta + ...
            (abs_diff >= beta) .* (abs_diff - 0.5 * beta) ...
        );
end