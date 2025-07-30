function layers = cycleIKNetwork(inputDim, outputDim)
% cycleIKNetwork 构建一个 8 层 MLP 网络，匹配原始 CycleIK 架构
% 输入:
%   inputDim: 输入维度（如 42）
%   outputDim: 输出维度（如 20）
% 输出:
%   layers: layer array，用于 dlnetwork 构建

layers = [
    featureInputLayer(inputDim, 'Normalization', 'none', 'Name', 'input')

    fullyConnectedLayer(2048, 'Name', 'fc1', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu1')

    fullyConnectedLayer(2048, 'Name', 'fc2', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu2')

    fullyConnectedLayer(1024, 'Name', 'fc3', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu3')

    fullyConnectedLayer(1024, 'Name', 'fc4', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu4')

    fullyConnectedLayer(512, 'Name', 'fc5', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu5')

    fullyConnectedLayer(512, 'Name', 'fc6', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu6')

    fullyConnectedLayer(256, 'Name', 'fc7', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu7')

    fullyConnectedLayer(256, 'Name', 'fc8', 'WeightsInitializer', 'he')
    geluLayer('Name', 'gelu8')

    fullyConnectedLayer(outputDim, 'Name', 'fc_out', 'WeightsInitializer', 'he')
    tanhLayer('Name', 'tanh_out')  % 将输出限制在 [-1, 1]
];
end
