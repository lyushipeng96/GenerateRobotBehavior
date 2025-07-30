function checkUnpairedFiles(inputFolder, outputFolder)
% 检查 Input 和 Output 文件夹下的 CSV 文件是否一一对应
% 输入:
%   inputFolder  - 'data/Input'
%   outputFolder - 'data/Output'

    % 获取文件列表
    inputFiles = dir(fullfile(inputFolder, 'Input_*.csv'));
    outputFiles = dir(fullfile(outputFolder, 'Output_*.csv'));

    % 提取标准化文件名编号部分（如 Input_00012 → 00012）
    inputIDs = extractBetween({inputFiles.name}, 'Input_', '.csv');
    outputIDs = extractBetween({outputFiles.name}, 'Output_', '.csv');

    % 转为字符串数组
    inputIDs = string(inputIDs);
    outputIDs = string(outputIDs);

    % 找出没有配对的 Input 和 Output 文件
    unmatchedInput = setdiff(inputIDs, outputIDs);
    unmatchedOutput = setdiff(outputIDs, inputIDs);

    % 显示结果
    fprintf("✅ Input 文件总数: %d\n", numel(inputIDs));
    fprintf("✅ Output 文件总数: %d\n", numel(outputIDs));
    fprintf("✅ 匹配成功的样本对: %d\n", numel(intersect(inputIDs, outputIDs)));

    if ~isempty(unmatchedInput)
        fprintf("\n❌ 以下 Input 文件没有对应的 Output：\n");
        disp("Missing Output for: " + "Input_" + unmatchedInput + ".csv");
    end

    if ~isempty(unmatchedOutput)
        fprintf("\n❌ 以下 Output 文件没有对应的 Input：\n");
        disp("Missing Input for: " + "Output_" + unmatchedOutput + ".csv");
    end
end
