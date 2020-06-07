clear all


load in_data %input data: m*n matrix, where m is the number of samples, n is the number of Aps (publie MAC addresses)
load out_data %output data: m*2 matrix, where m is the number of samples, 2 columns represents x and y axis of position coordinate

load results_90

rand_index = randperm(size(in_data,1));
num_train = round(0.75 * size(in_data,1));
num_valid = round(0.05 * size(in_data,1));
num_test = round(0.2 * size(in_data,1));

index_train = rand_index(1:num_train);
index_valid = rand_index(num_train + 1:num_train + num_valid);
index_test = rand_index(num_train + num_valid + 1:size(in_data,1));

train_Data = [];
target_Data = [];
test_Data = [];
target_test_Data = [];

%% manually create training and testing dataset (obsolete)
for i = 1:size(index_train,2)
    train_Data = [train_Data; Data_in(index_train(1,i),:)];
end

for i = 1:size(index_train,2)
    target_Data = [target_Data; Data_out(index_train(1,i),:)];
end

for i = 1:size(index_test,2)
    test_Data = [test_Data; Data_in(index_test(1,i),:)];
end

for i = 1:size(index_test,2)
    target_test_Data = [target_test_Data; Data_out(index_test(1,i),:)];
end


%% cerate multi-variable linear regression model for indoor dataset and plot Emperical CDF

beta1 = mvregress(train_Data,target_Data);

test_out = zeros(size(target_test_Data,1),size(target_test_Data,2));

test_out = test_Data * beta1;

raw_error = abs(target_test_Data - test_out);
test_error = raw_error';

test_error = [test_error; sqrt((test_error(1,:)).^2 + (test_error(2,:)).^2)];
sort_error = sortrows(test_error(3,:)')';

%%X = cdfplot(sort_error);

X = plot(sort_error,(1:size(sort_error,2))/length(sort_error));
hold on
set(X,'LineWidth',2);
xlabel('Absolute error [m]');
ylabel('CDF');
%title('Emperical CDF');
set(gca,'FontSize',14);
grid on


%% try deep neural network on indoor dataset and plot CDF graph.
net = fitnet([10 5 3]);
net.divideParam.trainRatio = 0.75;
net.divideParam.valRatio = 0.05;
net.divideParam.testRatio = 0.2;
[net train_info] = train(net,Data_in,Data_out);

test_index = train_info.testInd;

for i = 1:size(test_index,2)
    test_Data = [test_Data Data_in(:,test_index(i))];
    target_test_Data = [target_test_Data Data_out(:,test_index(i))];
end


net_out = net(test_Data);
error_1 = abs(target_test_Data - net_out); 
test_error2 = sqrt((error_1(1,:)).^2 + (error_1(2,:)).^2);
sort_error2 = sortrows(test_error2')';

X = plot(sort_error2,(1:size(sort_error2,2))/length(sort_error2));

hold on
set(X,'LineWidth',2);
grid on

 

%% legend for all three methods

legend({'Linear regression','Neural network'},'Location','SouthEast');