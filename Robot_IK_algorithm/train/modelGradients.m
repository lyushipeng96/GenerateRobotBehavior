% function [loss, gradients] = modelGradients(net, dlX, robotParams, cfg)
%     % 前向预测关节角度
%     predAngles = forward(net, dlX);    % 20×B
%     predAngles = tanh(predAngles);     % 角度范围 [-1, 1] 归一化（如需）
% 
%     % 反归一化到真实关节角
%     predAngles = denormalizeAngles(predAngles); % 20×B
% 
%     % 基于FK计算末端位姿
%     batchSize = size(predAngles, 2);
%     predPose = zeros(cfg.inputDim, batchSize, 'like', dlX);
%     % for b = 1:batchSize
%     %     predPose(:,b) = forwardKinematics(predAngles(:,b)', robotParams)';
%     % end
%     for b = 1:batchSize
%         angles_b = extractdata(predAngles(:,b)); % 20×1 普通矩阵
%         predPose(:,b) = forwardKinematics(angles_b', robotParams)'; % 42×1
%     end
% 
%     predPose = dlarray(predPose, 'CB');
% 
%     % 计算FK-loss
%     loss = cycleLoss(predPose, dlX, cfg);
% 
%     % 反向传播
%     gradients = dlgradient(loss, net.Learnables);
% end
function [loss, gradients] = modelGradients(net, dlX, robotParams, cfg)
% modelGradients 计算FK-loss与梯度（GPU兼容）
% 输入:
%   net         - dlnetwork
%   dlX         - 42×B dlarray (输入末端位姿)
%   robotParams - 机器人参数
%   cfg         - 配置文件
% 输出:
%   loss        - FK-loss
%   gradients   - 网络梯度

    % 前向预测关节角度（20×B）
    predAngles = forward(net, dlX);    
    predAngles = tanh(predAngles);        % 归一化 [-1, 1]
    predAngles = denormalizeAngles(predAngles); % 反归一化 (20×B)

    % 可微 FK 计算末端位姿 (42×B)
    predPose = forwardKinematics(predAngles, robotParams);

    % 计算 FK-loss
    loss = cycleLoss(predPose, dlX, cfg);

    % 自动求梯度
    gradients = dlgradient(loss, net.Learnables);
end
