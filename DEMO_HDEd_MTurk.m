load('DATA.mat');

fprintf('Demonstrating the training process and plot the learned model...\n');
scale = [-.5, .5];
resolution = 7;
% window_length = 1; % for demonstration
% [X, Y] = prepareDataHDEd(X_cont_delta, Label, scale, resolution, window_length);
% W = HDEd_train(X, Y, 0.0001, 0, 1);
% for t=1:length(W)
%     figure;
%     plotMaps(W{t}, resolution, 8, 16);
% end
% pause(1);

window_length = 3; % for better performance

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nstatic spectral contrast features, resolution=7, window length=3...\n');
Model = TrainHDEd_MTurk(X_cont_static, Label, RP, scale, resolution, window_length);
Result = TestHDEd(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\ndynamic spectral contrast features, resolution=7, window length=3...\n');
Model = TrainHDEd_MTurk(X_cont_delta, Label, RP, scale, resolution, window_length);
Result = TestHDEd(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nstatic MFCC features, resolution=7, window length=3...\n');
Model = TrainHDEd_MTurk(X_mfcc_static, Label, RP, scale, resolution, window_length);
Result = TestHDEd(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\ndynamic MFCC features, resolution=7, window length=3...\n');
Model = TrainHDEd_MTurk(X_mfcc_delta, Label, RP, scale, resolution, window_length);
Result = TestHDEd(Model)