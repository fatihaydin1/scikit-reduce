%% Run the reference MATLAB code (via Octave) on all prepared inputs.
args = argv(); inDir = args{1}; outDir = args{2};
alphas = [0.3 0.5 1 2];
files = dir(fullfile(inDir, 'nis_*.mat'));
for k = 1:numel(files)
    S = load(fullfile(inDir, files(k).name));
    R = struct();
    for a = 1:numel(alphas)
        R.(sprintf('idx_%d', a)) = sort(NIS_oct(S.X, alphas(a)));
    end
    save('-mat7-binary', fullfile(outDir, files(k).name), '-struct', 'R');
end
lambdas = [0.5 1 2];
files = dir(fullfile(inDir, 'bldod_*.mat'));
for k = 1:numel(files)
    S = load(fullfile(inDir, files(k).name));
    R = struct();
    for a = 1:numel(lambdas)
        R.(sprintf('score_%d', a)) = BLDOD_oct(S.X, lambdas(a));
    end
    save('-mat7-binary', fullfile(outDir, files(k).name), '-struct', 'R');
end
disp('done');
