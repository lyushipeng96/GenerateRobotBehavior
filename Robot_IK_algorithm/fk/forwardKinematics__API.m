function FeatureVector = forwardKinematics_API(jointAngles, Robot)
% forwardKinematics 使用 rigidBodyTree 进行 FK 计算
% 输入:
%   jointAngles - 1xN 向量（列向量形式）
%   robot       - rigidBodyTree 对象（由 importrobot 加载）
% 输出:
%   FeatureVector        - 42维 笛卡尔空间位姿 [P,R]
    jointAngles = double(jointAngles);
    JointNames = [
    "waist_roll", "head_yaw", "head_pitch", ...
    "l_upper_arm_roll", "l_lower_arm_pitch", "l_hand_pitch", ...
    "r_upper_arm_roll", "r_lower_arm_pitch", "r_hand_pitch"
    ];
    Position = zeros(10,3);
    Rotation = zeros(3,4);

    % 确保输入为列向量（robot.DataFormat == 'column'）
    % if isrow(jointAngles)
    %     jointAngles = jointAngles';
    % end
    
    Config = homeConfiguration(Robot);
    for i =13:32
        Config(i).JointPosition = jointAngles(i-12);
    end
    % 获取末端执行器的变换矩阵
    for i = 1:9
        tform = getTransform(Robot, Config, JointNames(i), 'base');
        Position(i+1,:) = tform(1:3, 4)';
        
        if mod(i, 3) == 0
            Rotation(i/3, :) = rotm2quat(tform(1:3,1:3));
        end
    end
    Position = reshape(Position', [1, 30]);
    Rotation = reshape(Rotation', [1, 12]);
    FeatureVector = [Position, Rotation];
end
