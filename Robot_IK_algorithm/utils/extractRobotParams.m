function robotParams = extractRobotParams(robot)
    numJoints = numel(robot.Bodies);
    robotParams.offset = zeros(4, 4, numJoints);
    robotParams.axis = zeros(numJoints, 3);
    robotParams.jointType = strings(1, numJoints);
    robotParams.bodyNames = strings(1, numJoints);

    for i = 1:numJoints
        body = robot.Bodies{i};
        joint = body.Joint;

        robotParams.offset(:,:,i) = joint.JointToParentTransform;
        robotParams.axis(i,:)     = joint.JointAxis;
        robotParams.jointType(i)  = joint.Type;
        robotParams.bodyNames(i)  = body.Name;
    end

    % Marker 名称
    robotParams.markerNames = [
    "waist_roll", "head_yaw", "head_pitch", ...
    "l_upper_arm_roll", "l_lower_arm_pitch", "l_hand_pitch", ...
    "r_upper_arm_roll", "r_lower_arm_pitch", "r_hand_pitch"
    ];

    % Marker 映射
    robotParams.markerMap = zeros(1, numel(robotParams.markerNames));
    for m = 1:numel(robotParams.markerNames)
        idx = find(strcmp(robotParams.bodyNames, robotParams.markerNames(m)), 1);
        if isempty(idx)
            error("Marker '%s' 在机器人模型中未找到！", robotParams.markerNames(m));
        end
        robotParams.markerMap(m) = idx;
    end
end
