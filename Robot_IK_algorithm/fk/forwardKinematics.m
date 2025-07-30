function FeatureVector = forwardKinematics(jointAngles, robotParams)
% forwardKinematics 基于 dlarray 的可微 FK 计算
% 输入:
%   jointAngles  - 20×B dlarray (控制关节角度)
%   robotParams  - extractRobotParams 输出的机器人参数
% 输出:
%   FeatureVector - 42×B dlarray [P(10*3), R(3*4)]

    totalJoints = size(robotParams.offset, 3);      % 所有关节
    controlledJoints = size(jointAngles, 1);        % 可控关节
    batchSize = size(jointAngles, 2);               % batch 大小
    numMarkers = numel(robotParams.markerMap);      % marker 数量

    % 获取 jointAngles 的底层类型 (single 或 double)
    baseType = underlyingType(jointAngles);

    % 初始化输出 (dlarray)
    Position = dlarray(zeros((numMarkers+1)*3, batchSize, baseType));
    Rotation = dlarray(zeros(3*4, batchSize, baseType));

    % 遍历 batch
    for b = 1:batchSize
        % 初始化变换矩阵
        T = dlarray(eye(4, baseType));
        angleIndex = 1;

        % 初始化 marker 存储
        pos = dlarray(zeros(numMarkers+1, 3, baseType));
        rot = dlarray(zeros(3, 4, baseType));

        % 遍历所有关节
        for j = 1:totalJoints
            % 固定偏移 (cast 并包装为 dlarray)
            T = T * dlarray(cast(robotParams.offset(:,:,j), baseType));

            % 关节角度
            if angleIndex <= controlledJoints
                theta = jointAngles(angleIndex, b);
            else
                theta = dlarray(cast(0, baseType)); % ✅ 修复
            end

            % 旋转或平移
            if robotParams.jointType(j) == "revolute"
                R = axisAngleRotation_dl(robotParams.axis(j,:), theta);
                jointTransform = dlarray(eye(4, baseType));
                jointTransform(1:3,1:3) = R;
                if angleIndex <= controlledJoints
                    angleIndex = angleIndex + 1;
                end
            elseif robotParams.jointType(j) == "prismatic"
                jointTransform = dlarray(eye(4, baseType));
                jointTransform(1:3,4) = robotParams.axis(j,:)' * theta;
                if angleIndex <= controlledJoints
                    angleIndex = angleIndex + 1;
                end
            else
                jointTransform = dlarray(eye(4, baseType)); % 固定关节
            end

            % 累积变换
            T = T * jointTransform;

            % 如果是 marker，记录位置与旋转
            markerIdx = find(robotParams.markerMap == j);
            if ~isempty(markerIdx)
                pos(markerIdx+1,:) = T(1:3, 4)'; % 位置
                if mod(markerIdx, 3) == 0
                    rot(markerIdx/3,:) = rotm2quat_dl(T(1:3,1:3));
                end
            end
        end

        % 展平
        Position(:,b) = reshape(pos', [], 1);   % 30×1
        Rotation(:,b) = reshape(rot', [], 1);   % 12×1
    end

    % 拼接 42 维特征
    FeatureVector = [Position; Rotation];
end

%% Rodrigues 旋转 (可微)
function R = axisAngleRotation_dl(axis, theta)
    axis = axis ./ norm(axis);
    x = axis(1); y = axis(2); z = axis(3);
    c = cos(theta); s = sin(theta); C = 1-c;

    R = [x*x*C+c,   x*y*C-z*s, x*z*C+y*s;
         y*x*C+z*s, y*y*C+c,   y*z*C-x*s;
         z*x*C-y*s, z*y*C+x*s, z*z*C+c];
end

%% 旋转矩阵转四元数 (可微)
function q = rotm2quat_dl(R)
    tr = R(1,1) + R(2,2) + R(3,3);
    if tr > 0
        S = sqrt(tr+1.0) * 2;
        qw = 0.25 * S;
        qx = (R(3,2) - R(2,3)) / S;
        qy = (R(1,3) - R(3,1)) / S;
        qz = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;
        qw = (R(3,2) - R(2,3)) / S;
        qx = 0.25 * S;
        qy = (R(1,2) + R(2,1)) / S;
        qz = (R(1,3) + R(3,1)) / S;
    elseif (R(2,2) > R(3,3))
        S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;
        qw = (R(1,3) - R(3,1)) / S;
        qx = (R(1,2) + R(2,1)) / S;
        qy = 0.25 * S;
        qz = (R(2,3) + R(3,2)) / S;
    else
        S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;
        qw = (R(2,1) - R(1,2)) / S;
        qx = (R(1,3) + R(3,1)) / S;
        qy = (R(2,3) + R(3,2)) / S;
        qz = 0.25 * S;
    end
    q = [qx; qy; qz; qw];
end
