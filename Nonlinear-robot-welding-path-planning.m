clc; clear; close all;

%% 参数设置
input_file = "D:\BY\QXGH\point_cloud_00001_s.ply";
output_dir = "D:\BY\QXGH\point_cloud_00001_sss";
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

z_offsets = [1.5, 1.5, 1.5];
num_slices = 100;
minDistance = 2.0;

region_params = [
    1/3, 0.25, 2;
    2/3, 0.34, 2;
];

ptCloud_full = pcread(input_file);
points_full = ptCloud_full.Location;

%% 计算点云质心作为旋转中心
center_point = mean(points_full, 1); % 计算点云质心 [Cx, Cy, Cz]

% 修改为绕X轴旋转180度（Y和Z坐标取反）
flip_point_set = @(pts) [pts(:,1), ...                    % X不变
                        2*center_point(2) - pts(:,2), ... % Y翻转
                        2*center_point(3) - pts(:,3)];    % Z翻转

z_vals_full = points_full(:,3);
max_z = max(z_vals_full);
min_z = min(z_vals_full);
z_range = max_z - min_z;

%% 可视化1：原始点云（绕X轴旋转后）
figure(1); clf;
flipped_full = flip_point_set(points_full);
pcshow(flipped_full, 'MarkerSize', 10, 'Background', 'w');
title('原始点云（绕X轴旋转180°）', 'FontSize', 14); 
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)'); 
grid on; axis equal;
saveas(gcf, fullfile(output_dir, '1_original_point_cloud_flipped.png'));

%% 顶部轨迹提取
z_threshold = max_z - 1.3;
top_points = points_full(z_vals_full > z_threshold, :);

%% 可视化2：顶部区域点云提取
figure(2); clf;
scatter3(flipped_full(:,1), flipped_full(:,2), flipped_full(:,3), 5, 'b', 'filled'); hold on;
flipped_top = flip_point_set(top_points);
scatter3(flipped_top(:,1), flipped_top(:,2), flipped_top(:,3), 20, 'r', 'filled');
title('顶部区域点云提取', 'FontSize', 14); 
legend({'全部点云', '顶部区域'}, 'Location', 'best');
grid on; axis equal; view(3);
saveas(gcf, fullfile(output_dir, '2_top_region_extraction.png'));

x_edges = linspace(min(top_points(:,1)), max(top_points(:,1)), num_slices+1);
trajectory_simple = zeros(num_slices, 3); 
valid_slices = 0;

for i = 1:num_slices
    in_slice = top_points(:,1) >= x_edges(i) & top_points(:,1) < x_edges(i+1);
    slice_points = top_points(in_slice, :);
    if ~isempty(slice_points)
        valid_slices = valid_slices + 1;
        trajectory_simple(valid_slices, :) = mean(slice_points, 1);
    end
end
trajectory_simple = trajectory_simple(1:valid_slices, :);

original_top_trajectory = trajectory_simple;
trajectory_simple(:,3) = trajectory_simple(:,3) - 3.0;

allTrajectories = {};
allTrajectoryPoints = {};
trajectoryCount = 1;
allTrajectories{trajectoryCount} = trajectory_simple;
allTrajectoryPoints{trajectoryCount} = top_points;
writematrix(trajectory_simple, fullfile(output_dir, 'trajectory_top_simple.csv'));

%% 区域聚类轨迹提取
figure(3); clf;
set(gcf, 'Position', [100, 100, 1200, 500]);

for regionIdx = 1:size(region_params,1)
    region_pos = region_params(regionIdx, 1);
    thickness = region_params(regionIdx, 2);
    expected_trajectories = region_params(regionIdx, 3);

    z_target = max_z - z_range * region_pos;
    region_mask = (z_vals_full > (z_target - thickness)) & (z_vals_full < (z_target + thickness));
    region_points = points_full(region_mask, :);
    ptCloud_region = pointCloud(region_points);

    [labels, numClusters] = pcsegdist(ptCloud_region, minDistance);
    found = 0;
    
    %% 可视化3：区域聚类结果
    subplot(1, size(region_params,1), regionIdx);
    cluster_colors = lines(numClusters);
    for clusterIdx = 1:numClusters
        idx = (labels == clusterIdx);
        cluster_pts = flip_point_set(region_points(idx, :));
        scatter3(cluster_pts(:,1), cluster_pts(:,2), cluster_pts(:,3), 20, ...
            cluster_colors(clusterIdx,:), 'filled'); hold on;
    end
    title(sprintf('区域%d聚类结果 (%d个簇)', regionIdx, numClusters), 'FontSize', 12);
    grid on; axis equal; view(-30,30);
    
    for clusterIdx = 1:numClusters
        idx = (labels == clusterIdx);
        cluster_points = region_points(idx, :);
        if isempty(cluster_points), continue; end

        z_vals = cluster_points(:,3);
        max_z_cluster = max(z_vals);
        z_thresh = max_z_cluster - z_offsets(min(regionIdx, numel(z_offsets)));
        high_points = cluster_points(z_vals > z_thresh, :);
        if size(high_points,1) < 10, continue; end

        x_edges = linspace(min(high_points(:,1)), max(high_points(:,1)), num_slices+1);
        trajectory = zeros(num_slices, 3); 
        valid_points = 0;
        
        for i = 1:num_slices
            in_slice = high_points(:,1) >= x_edges(i) & high_points(:,1) < x_edges(i+1);
            slice_points = high_points(in_slice, :);
            if ~isempty(slice_points)
                valid_points = valid_points + 1;
                trajectory(valid_points, :) = mean(slice_points, 1);
            end
        end
        
        if valid_points < 2, continue; end
        trajectory = trajectory(1:valid_points, :);
        
        trajectory(:,3) = trajectory(:,3) - 3.0;

        trajectoryCount = trajectoryCount + 1;
        allTrajectories{trajectoryCount} = trajectory;
        allTrajectoryPoints{trajectoryCount} = high_points;

        writematrix(trajectory, fullfile(output_dir, ...
            sprintf('trajectory_r%d_c%d.csv', regionIdx, clusterIdx)));

        found = found + 1;
        if found >= expected_trajectories
            break;
        end
    end
end
saveas(gcf, fullfile(output_dir, '3_region_clustering_results.png'));

%% 可视化4：轨迹提取过程
figure(4); clf;
set(gcf, 'Position', [100, 100, 1000, 800]);

% 获取一组颜色
color_set = lines(7); % 获取7种不同的颜色

% 顶部轨迹
subplot(2,2,1);
traj_flipped = flip_point_set(trajectory_simple);
plot3(traj_flipped(:,1), traj_flipped(:,2), traj_flipped(:,3), ...
    'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
title('顶部轨迹提取', 'FontSize', 12);
grid on; axis equal; view(3);

% 各区域轨迹
for regionIdx = 1:min(3, length(allTrajectories)-1)
    subplot(2,2,regionIdx+1);
    traj_flipped = flip_point_set(allTrajectories{regionIdx+1});
    
    % 使用不同的颜色 - 修复错误
    current_color = color_set(regionIdx+1, :); % 获取RGB三元组
    
    plot3(traj_flipped(:,1), traj_flipped(:,2), traj_flipped(:,3), ...
        'o-', 'Color', current_color, 'LineWidth', 2, 'MarkerFaceColor', current_color);
    title(sprintf('区域%d轨迹提取', regionIdx), 'FontSize', 12);
    grid on; axis equal; view(3);
end
saveas(gcf, fullfile(output_dir, '4_trajectory_extraction_process.png'));

%% 插入顶部轨迹副本至 2/3 高度
middle_traj = original_top_trajectory;

target_z_middle = max_z - z_range * (2/3);
mean_z_orig = mean(middle_traj(:,3));
z_shift = target_z_middle - mean_z_orig;
middle_traj(:,3) = middle_traj(:,3) + z_shift;
middle_traj(:,3) = middle_traj(:,3) - 3.0;

trajectoryCount = trajectoryCount + 1;
allTrajectories{trajectoryCount} = middle_traj;
allTrajectoryPoints{trajectoryCount} = top_points;

writematrix(middle_traj, fullfile(output_dir, 'trajectory_middle_inserted.csv'));

%% 可视化5：中间层轨迹插入
figure(5); clf;
original_top_flipped = flip_point_set(original_top_trajectory);
middle_traj_flipped = flip_point_set(middle_traj);

scatter3(original_top_flipped(:,1), original_top_flipped(:,2), original_top_flipped(:,3), ...
    50, 'r', 'filled'); hold on;
scatter3(middle_traj_flipped(:,1), middle_traj_flipped(:,2), middle_traj_flipped(:,3), ...
    50, 'b', 'filled');

% 添加高度标注
text(mean(original_top_flipped(:,1)), mean(original_top_flipped(:,2)), max(original_top_flipped(:,3))+10, ...
    sprintf('原始高度: %.1fmm', mean(original_top_flipped(:,3))), 'FontSize',12, 'Color', 'k');
text(mean(middle_traj_flipped(:,1)), mean(middle_traj_flipped(:,2)), mean(middle_traj_flipped(:,3))-10, ...
    sprintf('插入高度: %.1fmm', mean(middle_traj_flipped(:,3))), 'FontSize',12, 'Color', 'k');

title('中间层轨迹插入', 'FontSize', 14); 
legend({'原始顶部轨迹', '插入的中间层轨迹'}, 'Location', 'best');
grid on; axis equal; view(3);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
saveas(gcf, fullfile(output_dir, '5_middle_layer_insertion.png'));

%% 构建完整焊接轨迹（确保正确拼接）
fullTrajectory = allTrajectories{1};
layer_indices = ones(size(allTrajectories{1},1),1);

current_idx = 1;
first_traj = allTrajectories{1};
n1 = size(first_traj, 1);
fullTrajectory(current_idx:current_idx+n1-1, :) = first_traj;
layer_indices(current_idx:current_idx+n1-1) = 1;
current_idx = current_idx + n1;

for i = 2:length(allTrajectories)
    traj = allTrajectories{i};
    % 判断轨迹是否需要反转，确保连接方向正确
    if norm(fullTrajectory(current_idx-1,:) - traj(end,:)) < norm(fullTrajectory(current_idx-1,:) - traj(1,:))
        traj = flipud(traj);  % 翻转轨迹
    end
    
    n = size(traj, 1);
    fullTrajectory(current_idx:current_idx+n-1, :) = traj;
    layer_indices(current_idx:current_idx+n-1) = i;
    current_idx = current_idx + n;
end

%% 层间连接点生成
% 计算轨迹点平均间距
total_distance = 0;
for i = 1:(size(fullTrajectory,1)-1)
    total_distance = total_distance + norm(fullTrajectory(i+1,:) - fullTrajectory(i,:));
end
avg_spacing = total_distance / (size(fullTrajectory,1)-1);
connection_spacing = 3 * avg_spacing; % 层间连接点间距

% 将完整轨迹按层分割
segments = {};
current_layer = layer_indices(1);
start_idx = 1;
for i = 2:length(layer_indices)
    if layer_indices(i) ~= current_layer
        segments{end+1} = fullTrajectory(start_idx:i-1, :);
        start_idx = i;
        current_layer = layer_indices(i);
    end
end
segments{end+1} = fullTrajectory(start_idx:end, :);

%% 可视化6：轨迹分段与连接点
figure(6); clf;
set(gcf, 'Position', [100, 100, 1000, 800]);

% 获取一组颜色
color_set = lines(7); % 获取7种不同的颜色

% 绘制各段原始轨迹
for i = 1:length(segments)
    seg_flipped = flip_point_set(segments{i});
    
    % 使用不同的颜色 - 循环使用颜色集
    current_color = color_set(mod(i-1, size(color_set,1)) + 1, :); 
    
    plot3(seg_flipped(:,1), seg_flipped(:,2), seg_flipped(:,3), 'o-', ...
        'Color', current_color, 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', current_color); 
    hold on;
end

% 绘制连接点
for seg = 1:length(segments)-1
    p1 = flip_point_set(segments{seg}(end,:));
    p2 = flip_point_set(segments{seg+1}(1,:));
    
    % 计算连接点
    vec = p2 - p1;
    dist = norm(vec);
    num_insertions = ceil(dist / connection_spacing) - 1;
    if num_insertions < 0, num_insertions = 0; end
    
    for k = 1:num_insertions
        t = k / (num_insertions + 1);
        interp_point = p1 + t * vec;
        plot3(interp_point(1), interp_point(2), interp_point(3), 'ks', ...
            'MarkerSize', 10, 'MarkerFaceColor', 'y');
    end
end

title('轨迹分段与层间连接点', 'FontSize', 14); 
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)'); 
grid on; axis equal; view(3);

% 创建图例
legend_entries = arrayfun(@(i) sprintf('段 %d', i), 1:length(segments), 'UniformOutput', false);
legend_entries{end+1} = '连接点';
legend(legend_entries, 'Location', 'best');

saveas(gcf, fullfile(output_dir, '6_trajectory_connection_process.png'));

% 在层间插入连接点
connectedTrajectory = segments{1}; % 从第一个点开始
for seg = 1:length(segments)-1
    % 当前层终点
    p1 = segments{seg}(end,:);
    % 下一层起点
    p2 = segments{seg+1}(1,:);
    
    % 计算方向和距离
    vec = p2 - p1;
    dist = norm(vec);
    dir_vec = vec / dist;
    
    % 计算需要插入的点数
    num_insertions = ceil(dist / connection_spacing) - 1;
    if num_insertions < 0, num_insertions = 0; end
    
    % 生成并添加连接点
    for k = 1:num_insertions
        t = k / (num_insertions + 1);
        interp_point = p1 + t * vec;
        connectedTrajectory = [connectedTrajectory; interp_point];
    end
    
    % 添加下一层轨迹
    connectedTrajectory = [connectedTrajectory; segments{seg+1}];
end

%% 起点与终点的定义
% 在终点正上方创建新点
newPoint = connectedTrajectory(end, :);
newPoint(1) = newPoint(1) - 20;
newPoint(2) = newPoint(2) - 10;  % Y方向偏移10mm
newPoint(3) = newPoint(3) - 12;  % Z方向偏移-12mm

% 计算起点到轨迹起点、轨迹终点到终点的连接点
p0 = newPoint; % 起点
p1 = connectedTrajectory(1,:); % 轨迹起点
p2 = connectedTrajectory(end,:); % 轨迹终点
p3 = newPoint; % 终点

% 计算起点到轨迹起点的连接点
vec01 = p1 - p0;
dist01 = norm(vec01);
num_insert_pre = max(ceil(dist01 / connection_spacing) - 1, 0);

% 计算轨迹终点到终点的连接点
vec23 = p3 - p2;
dist23 = norm(vec23);
num_insert_post = max(ceil(dist23 / connection_spacing) - 1, 0);

% 构建增强轨迹
enhancedTrajectory = p0; % 起点

% 插入起点到轨迹起点的中间点
for k = 1:num_insert_pre
    t = k / (num_insert_pre + 1);
    interp_point = p0 + t * vec01;
    enhancedTrajectory = [enhancedTrajectory; interp_point];
end

% 加入整个轨迹
enhancedTrajectory = [enhancedTrajectory; connectedTrajectory];

% 插入轨迹终点到终点的中间点
for k = 1:num_insert_post
    t = k / (num_insert_post + 1);
    interp_point = p2 + t * vec23;
    enhancedTrajectory = [enhancedTrajectory; interp_point];
end

% 加入终点
enhancedTrajectory = [enhancedTrajectory; p3];

% 对完整轨迹应用绕X轴旋转180度
enhancedTrajectory_display = flip_point_set(enhancedTrajectory);

%% 可视化7：起点终点增强处理
figure(7); clf;
set(gcf, 'Position', [100, 100, 1000, 800]);

% 绘制完整轨迹
plot3(enhancedTrajectory_display(:,1), enhancedTrajectory_display(:,2), ...
    enhancedTrajectory_display(:,3), 'b-', 'LineWidth', 1.5); hold on;

% 标记特殊点
plot3(enhancedTrajectory_display(1,1), enhancedTrajectory_display(1,2), ...
    enhancedTrajectory_display(1,3), 'mo', 'MarkerSize', 14, 'MarkerFaceColor', 'm');
plot3(enhancedTrajectory_display(end,1), enhancedTrajectory_display(end,2), ...
    enhancedTrajectory_display(end,3), 'mo', 'MarkerSize', 14, 'MarkerFaceColor', 'm');

% 标注连接段
text(mean([enhancedTrajectory_display(1,1), enhancedTrajectory_display(2,1)]), ...
     mean([enhancedTrajectory_display(1,2), enhancedTrajectory_display(2,2)]), ...
     mean([enhancedTrajectory_display(1,3), enhancedTrajectory_display(2,3)]), ...
    '起点连接段', 'FontSize',12, 'Color','k', 'BackgroundColor', 'w');
text(mean([enhancedTrajectory_display(end-1,1), enhancedTrajectory_display(end,1)]), ...
     mean([enhancedTrajectory_display(end-1,2), enhancedTrajectory_display(end,2)]), ...
     mean([enhancedTrajectory_display(end-1,3), enhancedTrajectory_display(end,3)]), ...
    '终点连接段', 'FontSize',12, 'Color','k', 'BackgroundColor', 'w');

title('起点/终点增强处理', 'FontSize', 14);
legend({'焊接轨迹', '起点/终点'}, 'Location', 'best');
grid on; axis equal; view(3);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
saveas(gcf, fullfile(output_dir, '7_start_end_enhancement.png'));

% 保存显示坐标系下的增强轨迹（旋转后）
output_file = fullfile(output_dir, 'full_welding_trajectory.txt');
fid = fopen(output_file, 'w');
fprintf(fid, 'X_mm\tY_mm\tZ_mm\n'); 
for i = 1:size(enhancedTrajectory_display,1)
    fprintf(fid, '%.6f\t%.6f\t%.6f\n', enhancedTrajectory_display(i,:));
end
fclose(fid);

%% 可视化8：轨迹点间距分析
figure(8); clf;
set(gcf, 'Position', [100, 100, 1200, 500]);

% 计算相邻点间距
distances = vecnorm(diff(enhancedTrajectory_display), 2, 2);
subplot(1,2,1);
histogram(distances, 50, 'FaceColor', [0.2 0.6 0.8]);
title('轨迹点间距分布', 'FontSize', 14);
xlabel('间距 (mm)'); ylabel('点数');
grid on;

% 标记特殊点
hold on;
avg_line = xline(avg_spacing, 'r-', 'LineWidth', 2, 'DisplayName', '平均间距');
conn_line = xline(connection_spacing, 'g-', 'LineWidth', 2, 'DisplayName', '连接点间距');
legend([avg_line, conn_line]);

% 层间连接点间距分析
subplot(1,2,2);
conn_distances = distances(distances > connection_spacing*0.9);
if ~isempty(conn_distances)
    plot(conn_distances, 'bo-', 'LineWidth', 1.5);
    title('层间连接点间距', 'FontSize', 14);
    xlabel('连接点序号'); ylabel('间距 (mm)');
    yline(connection_spacing, 'r--', '目标间距');
    grid on;
else
    text(0.5, 0.5, '未检测到层间连接点', 'HorizontalAlignment','center', 'FontSize',14);
end
saveas(gcf, fullfile(output_dir, '8_trajectory_spacing_analysis.png'));

%% 可视化9：最终轨迹验证
figure(9); clf;
set(gcf, 'Position', [100, 100, 1200, 600]);

% 左侧：内存中的轨迹
subplot(1,2,1);
plot3(enhancedTrajectory_display(:,1), enhancedTrajectory_display(:,2), ...
    enhancedTrajectory_display(:,3), 'b-o', 'MarkerFaceColor','b', 'MarkerSize',4);
hold on;
plot3(enhancedTrajectory_display(1,1), enhancedTrajectory_display(1,2), ...
    enhancedTrajectory_display(1,3), 'mo', 'MarkerSize',14, 'MarkerFaceColor','m');
plot3(enhancedTrajectory_display(end,1), enhancedTrajectory_display(end,2), ...
    enhancedTrajectory_display(end,3), 'mo', 'MarkerSize',14, 'MarkerFaceColor','m');
title('内存中的轨迹', 'FontSize', 14);
grid on; axis equal; view(-30,30);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');

% 右侧：文件加载的轨迹
subplot(1,2,2);
saved_traj = readmatrix(output_file, 'NumHeaderLines', 1);
plot3(saved_traj(:,1), saved_traj(:,2), saved_traj(:,3), 'r-o', ...
    'MarkerFaceColor','r', 'MarkerSize',4);
hold on;
plot3(saved_traj(1,1), saved_traj(1,2), saved_traj(1,3), 'mo', ...
    'MarkerSize',14, 'MarkerFaceColor','m');
plot3(saved_traj(end,1), saved_traj(end,2), saved_traj(end,3), 'mo', ...
    'MarkerSize',14, 'MarkerFaceColor','m');
title('文件保存的轨迹', 'FontSize', 14);
grid on; axis equal; view(-30,30);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');

% 计算最大偏差
max_diff = max(vecnorm(enhancedTrajectory_display - saved_traj, 2, 2));
title_str = sprintf('轨迹一致性验证 | 最大偏差: %.6f mm', max_diff);
sgtitle(title_str, 'FontSize', 16, 'FontWeight', 'bold');

% 保存验证图
saveas(gcf, fullfile(output_dir, '9_final_trajectory_validation.png'));

%% 打印关键参数
fprintf('===== 轨迹参数汇总 =====\n');
fprintf('点云质心坐标: [%.2f, %.2f, %.2f] mm\n', center_point);
fprintf('轨迹点平均间距: %.4f mm\n', avg_spacing);
fprintf('层间连接点间距: %.4f mm\n', connection_spacing);
fprintf('总轨迹点数: %d\n', size(enhancedTrajectory,1));
fprintf('起点坐标: [%.2f, %.2f, %.2f] mm\n', enhancedTrajectory_display(1,:));
fprintf('终点坐标: [%.2f, %.2f, %.2f] mm\n', enhancedTrajectory_display(end,:));
fprintf('轨迹文件保存至: %s\n', output_file);

%% 可视化10：最终焊接轨迹在点云上的显示
figure(10); clf;
set(gcf, 'Position', [100, 100, 1200, 800]);

% 显示旋转后的点云
pcshow(flipped_full, 'MarkerSize', 10, 'Background', 'w'); hold on;

% 绘制增强轨迹
plot3(enhancedTrajectory_display(:,1), enhancedTrajectory_display(:,2), ...
    enhancedTrajectory_display(:,3), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'MarkerSize', 5);

% 标记起点和终点
plot3(enhancedTrajectory_display(1,1), enhancedTrajectory_display(1,2), ...
    enhancedTrajectory_display(1,3), 'mo', 'MarkerSize', 20, 'MarkerFaceColor', 'm');
plot3(enhancedTrajectory_display(end,1), enhancedTrajectory_display(end,2), ...
    enhancedTrajectory_display(end,3), 'mo', 'MarkerSize', 20, 'MarkerFaceColor', 'm');

title('最终焊接轨迹在点云上的显示', 'FontSize', 16);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
legend({'点云', '焊接轨迹', '起点/终点'}, 'Location', 'best');
grid on; axis equal; view(-30, 30);
saveas(gcf, fullfile(output_dir, '10_final_trajectory_on_point_cloud.png'));

figure(11);
I=pcread("point_cloud_00009_s.ply");
pcshow(I);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)'); 
grid on; axis equal;


fprintf('所有可视化已完成并保存到目录: %s\n', output_dir);