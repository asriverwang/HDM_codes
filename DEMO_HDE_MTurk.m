load('DATA.mat');

fprintf('Demonstrating the training process and plot the learned model...\n');
scale = [-.5, .5];
resolution = 7;
[X, Y] = prepareData(X_cont_delta, Label, scale, resolution);
W = HDE_train(X, Y, 0.0001, 0, 1);
plotMaps(W, resolution, 8, 16);
pause(1);

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nwith the static spectral contrast features, resolution=7...\n');
Model = TrainHDE_MTurk(X_cont_static, Label, RP, scale, resolution);
Result = TestHDE(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nwith the bundle spectral contrast features (w/ delta- and delta-delta),\nresolution=7...\n');
Model = TrainHDE_MTurk(X_cont_delta, Label, RP, scale, resolution);
Result = TestHDE(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nwith the static MFCC features, resolution=7...\n');
Model = TrainHDE_MTurk(X_mfcc_static, Label, RP, scale, resolution);
Result = TestHDE(Model)

fprintf('\n\nDemonstrating the train-test experiment (repeat 10 times)...\nwith the bundle MFCC features (w/ delta- and delta-delta), resolution=7...\n');
Model = TrainHDE_MTurk(X_mfcc_delta, Label, RP, scale, resolution);
Result = TestHDE(Model)