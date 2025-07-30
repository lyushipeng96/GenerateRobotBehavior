function trainCycleIK()
    addpath(genpath('model'));
    addpath(genpath('loss'));
    addpath(genpath('fk'));
    addpath(genpath('utils'));
    addpath(genpath('data'));

    % 配置
    cfg = config();
    [Robot, ~, ~] = robot_setting();
    robotParams = extractRobotParams(Robot);

    % 加载数据集
    load('data/train_data-1000-v1.mat', 'input_vec'); % N×42
    numSamples = size(input_vec, 1);

    % 数据划分 (8:2)
    idx = randperm(numSamples);
    trainSize = floor(0.8 * numSamples);
    trainData = input_vec(idx(1:trainSize), :);
    valData   = input_vec(idx(trainSize+1:end), :);

    % 网络初始化
    layers = cycleIKNetwork(cfg.inputDim, cfg.outputDim);
    net = dlnetwork(layerGraph(layers));

    % GPU
    if canUseGPU
        net = dlupdate(@gpuArray, net);
        trainData = gpuArray(single(trainData));
        valData   = gpuArray(single(valData));
    else
        trainData = single(trainData);
        valData   = single(valData);
    end

    % Adam 初始化
    trailingAvg = [];
    trailingAvgSq = [];
    learnRate = cfg.lr;

    % GUI: 初始化图形
    fig = figure('Name','CycleIK Training','NumberTitle','off');
    hold on;
    grid on;
    xlabel('Epoch');
    ylabel('Loss');
    title('Training vs Validation Loss');
    trainLossLine = animatedline('Color','b','LineWidth',1.5,'DisplayName','Train Loss');
    valLossLine   = animatedline('Color','r','LineWidth',1.5,'DisplayName','Val Loss');
    legend;

    % 训练
    for epoch = 1:cfg.epochs
        % ===== 训练 =====
        idxTrain = randperm(trainSize);
        trainData = trainData(idxTrain, :);
        totalTrainLoss = 0;
        numBatches = 0;

        for i = 1:cfg.batchSize:trainSize
            % i
            idxRange = i:min(i+cfg.batchSize-1, trainSize);
            X = dlarray(trainData(idxRange,:)', 'CB');
            if canUseGPU, X = gpuArray(X); end

            [loss, gradients] = dlfeval(@modelGradients, net, X, robotParams, cfg);
            gradients = dlupdate(@(g) max(min(g, cfg.gradClip), -cfg.gradClip), gradients);
            % 计算平均梯度值
            gradNorm = 0;
            for g = 1:height(gradients)
                gradVal = extractdata(gradients.Value{g});
                gradNorm = gradNorm + mean(abs(gradVal), 'all');
            end
            fprintf("Epoch %d | Mean Gradient Magnitude: %.6e\n", epoch, gradNorm);
            % Adam 更新
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
                trailingAvg, trailingAvgSq, i, learnRate);

            totalTrainLoss = totalTrainLoss + double(gather(loss));
            numBatches = numBatches + 1;
        end
        trainLoss = totalTrainLoss / numBatches;

        % ===== 验证 =====
        totalValLoss = 0;
        numValBatches = 0;
        for i = 1:cfg.batchSize:size(valData,1)
            idxRange = i:min(i+cfg.batchSize-1, size(valData,1));
            Xval = dlarray(valData(idxRange,:)', 'CB');
            if canUseGPU, Xval = gpuArray(Xval); end

            predAngles = forward(net, Xval);
            predAngles = tanh(predAngles);
            predAngles = denormalizeAngles(predAngles);

            predPose = forwardKinematics(predAngles, robotParams);
            lossVal = cycleLoss(predPose, Xval, cfg);

            totalValLoss = totalValLoss + double(gather(lossVal));
            numValBatches = numValBatches + 1;
        end
        valLoss = totalValLoss / numValBatches;
        % 更新学习率
        learnRate = max(learnRate - cfg.lrDecay, 1e-6);
        % 输出训练进度
        fprintf("Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f\n", ...
            epoch, cfg.epochs, trainLoss, valLoss);
        % GUI 更新
        addpoints(trainLossLine, epoch, trainLoss);
        addpoints(valLossLine, epoch, valLoss);
        drawnow;
    end
    save('trained_model.mat', 'net');
    disp('✅ 训练完成');
end