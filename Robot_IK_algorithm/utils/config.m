function cfg = config()
% config: 返回结构体，包含所有训练、网络和损失参数

    % 网络结构
    cfg.inputDim = 42;
    cfg.outputDim = 20;
    cfg.hiddenSizes = [2048 2048 1024 1024 512 512 256 256];
    cfg.activation = 'gelu';
    cfg.outputActivation = 'tanh';

    % 训练超参数
    cfg.batchSize = 128;
    cfg.lr = 1e-5;
    cfg.epochs = 2;
    cfg.gradClip = 1.0;
    % cfg.gradientClip = 1.0; % 开启梯度裁剪

    % 学习率调度
    % cfg.lrDecay = cfg.lr / cfg.epochs;
    cfg.lrDecay = 1e-6;   % 每步衰减

    % 损失函数权重
    cfg.w_pos = 10;
    cfg.w_rot = 5;
    cfg.beta_pos = 0.01;
    cfg.beta_rot = 0.05;
end