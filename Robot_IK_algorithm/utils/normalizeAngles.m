function normAngles = normalizeAngles(angles)
% 将关节角度归一化到 [-1, 1]
% 输入:  angles: Nx20 原始角度矩阵
% 输出:  normAngles: Nx20 映射到 [-1, 1]

    [lo, hi] = jointLimits();  % 1x20

    % 扩展为 Nx20，与 angles 对应
    normAngles = 2 * (angles - lo) ./ (hi - lo) - 1;
end
