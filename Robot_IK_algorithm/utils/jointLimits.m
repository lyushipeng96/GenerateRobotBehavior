function [lo, hi] = jointLimits()
% 所有关节角限制统一设为 [-pi, pi]
% 输出:
%   lo: 1x20 下限数组 (-pi)
%   hi: 1x20 上限数组 (+pi)

    lo = -pi * ones(1, 20);
    hi =  pi * ones(1, 20);
end
