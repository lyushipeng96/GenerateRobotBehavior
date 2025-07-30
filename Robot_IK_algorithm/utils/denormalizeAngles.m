function angles = denormalizeAngles(normAngles)
% 将网络输出 [-1, 1] 转换为关节角度 (dlarray 保持可微)
    [lo, hi] = jointLimits();

    % 确保 lo 和 hi 转为 dlarray 并广播到 batch
    if ~isdlarray(normAngles)
        error("denormalizeAngles: Input must be dlarray.");
    end

    lo = dlarray(single(lo(:))); % 20×1
    hi = dlarray(single(hi(:))); % 20×1

    % 处理 batch (广播)
    lo = repmat(lo, 1, size(normAngles, 2));
    hi = repmat(hi, 1, size(normAngles, 2));

    % 反归一化
    angles = 0.5 * (normAngles + 1) .* (hi - lo) + lo; % 20×B dlarray
end
