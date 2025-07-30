function [robot,currentRobotJConfig,Body_joint_PositionLimits] = robot_setting()
%ROBOT_SETTING 完成GR1机器人仿真初始化设置
%   Detailed explanation goes here
    robot= importrobot("GR1T1.urdf");
    % Markers1=readNPY('dance.npy');
    show(robot);
    axis off
    currentRobotJConfig = homeConfiguration(robot); % 获取当前关节配置
    numJoints = numel(currentRobotJConfig);       % 获取机器人自由度
    Body_joint_PositionLimits=zeros(numJoints,2);
    for i=1:numJoints
    %     Body_joint_PositionLimits(i,1)=robot.Bodies{1,i}.Joint.PositionLimits(1);
    %     Body_joint_PositionLimits(i,2)=robot.Bodies{1,i}.Joint.PositionLimits(2);
        Body_joint_PositionLimits(i,1)=-pi;
        Body_joint_PositionLimits(i,2)=pi;
    end
end

