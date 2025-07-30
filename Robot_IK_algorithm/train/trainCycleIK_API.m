function trainCycleIK_API()
    addpath(genpath('model'));
    addpath(genpath('loss'));
    addpath(genpath('fk'));
    addpath(genpath('utils'));
    addpath(genpath('data'));

    cfg = config();
    % robot = importrobot('GR1T1.urdf', 'DataFormat', 'column');
    [Robot,~,~] = robot_setting();
    load('data/train_data-1000-v1.mat', 'input_vec'); % 只需要输入 pose 数据

    layers = cycleIKNetwork(cfg.inputDim, cfg.outputDim);
    net = dlnetwork(layerGraph(layers));

    trailingAvg = [];
    trailingAvgSq = [];
    learnRate = cfg.lr;
    numSamples = size(input_vec, 1);

    for epoch = 1:cfg.epochs
        idx = randperm(numSamples);
        input_vec = input_vec(idx, :);

        for i = 1:cfg.batchSize:numSamples
            idxRange = i:min(i+cfg.batchSize-1, numSamples);
            X = dlarray(single(input_vec(idxRange, :)'), 'CB');

            % 使用 dlfeval 计算伪梯度
            [gradients] = dlfeval(@pseudoGradients, net, X);

            % 前向预测角度
            predAngles = extractdata(forward(net, X))';

            % 计算真实 FK-loss
            lossVal = cycleLoss(predAngles, input_vec(idxRange,:), Robot, cfg);

            % 用 FK-loss 缩放梯度
            gradients = dlupdate(@(g) g .* (lossVal / (1 + lossVal)), gradients);

            % Adam 更新
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
                trailingAvg, trailingAvgSq, i, learnRate);
        end

        learnRate = max(learnRate - cfg.lrDecay, 1e-6);
        fprintf("Epoch %d/%d completed. Loss: %.6f\n", epoch, cfg.epochs, lossVal);
    end

    save('trained_model.mat', 'net');
    disp('训练完成');
end

function gradients = pseudoGradients(net, dlX)
    % 一个伪梯度计算函数
    % 生成与网络参数相关的伪 loss，以便 dlgradient 返回梯度
    y = forward(net, dlX);
    dummyLoss = sum(y, 'all');  % 让 loss 与参数相关
    gradients = dlgradient(dummyLoss, net.Learnables);
end