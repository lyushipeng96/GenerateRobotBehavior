function [input_vec, target_angles] = loadPoseAngleDataset(inputFolder, outputFolder)
% 批量读取 Input/Output 下的 CSV 样本对，构建归一化后的训练集

    % 获取所有 Input/Output 文件
    inputFiles = dir(fullfile(inputFolder, 'Inout_*.csv'));
    outputFiles = dir(fullfile(outputFolder, 'Output_*.csv'));

    if length(inputFiles) ~= length(outputFiles)
        error("输入和输出文件数不一致！");
    end

    numSamples = length(inputFiles)
    % print(numSamples)
    input_vec = zeros(numSamples, 42);
    raw_angles = zeros(numSamples, 20);

    % for i = 1:numSamples
    for i = 1:numSamples
        inputPath = fullfile(inputFolder, inputFiles(i).name);
        outputPath = fullfile(outputFolder, outputFiles(i).name);

        input_vec(i, :) = readmatrix(inputPath);       % pose
        raw_angles(i, :) = readmatrix(outputPath);     % raw joint angles
        if mod(i,10000) == 0
            t=i/10000
        end
    end

    % 归一化关节角到 [-1, 1]，使其匹配 tanh 输出
    target_angles = normalizeAngles(raw_angles);

    % 保存 MAT 文件供训练使用（可选）
    save('data/train_data.mat', 'input_vec', 'target_angles');
end
