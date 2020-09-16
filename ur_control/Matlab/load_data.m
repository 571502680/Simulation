
clc;
close all;
clear;

name_prefix = 1;

data_path = '/home/zh/catkin_ws/src/move_test/record_servoj_sim/';
suffix_sim = '_servoj_tracking.txt';
suffix_real  = '_servoj_tracking_real_robot.txt';
path = [data_path , num2str(name_prefix), suffix_real];


result = load(path);
t = result(:,1);
z_ref = result(:,4);
z_actual = result(:,7);

plot(t,z_ref,'LineWidth',1.2);hold on;
plot(t,z_actual,  'LineWidth',1.2);
legend('Ref','real'); grid on; grid minor;
xlabel('time (s)');
ylabel('z_position(m)')
axis([0,20,-inf,inf])












