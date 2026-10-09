function export_llamafactory_dataset(datasetName, imageDir, imageExtension)
%EXPORT_LLAMAFACTORY_DATASET Export QA JSONL in LLaMA-Factory image format.
% Original JSONL files are preserved. Image paths are relative, for example
% `ocean/sample_000001.jpg`.

if nargin < 1 || isempty(datasetName), datasetName = 'underwater_5class_v1'; end
if nargin < 2 || isempty(imageDir), imageDir = 'ocean'; end
if nargin < 3 || isempty(imageExtension), imageExtension = '.jpg'; end
if imageExtension(1) ~= '.', imageExtension = ['.' imageExtension]; end
imageDir = strrep(imageDir, '\', '/');

root = fileparts(fileparts(mfilename('fullpath')));
datasetDir = fullfile(root, 'output', 'datasets', datasetName);
for split = {'train', 'val'}
    splitName = split{1};
    sourceFile = fullfile(datasetDir, [splitName '_qa.jsonl']);
    if exist(sourceFile, 'file') ~= 2, continue; end
    outputFile = fullfile(datasetDir, [splitName '_llamafactory.jsonl']);
    count = exportSplit(sourceFile, outputFile, imageDir, imageExtension);
    fprintf('%s: %d samples -> %s\n', splitName, count, outputFile);
end
end

function count = exportSplit(sourceFile, outputFile, imageDir, imageExtension)
lines = readlines(sourceFile);
lines = lines(strlength(strtrim(lines)) > 0);
id = fopen(outputFile, 'w');
assert(id ~= -1, 'Cannot write %s.', outputFile);
closer = onCleanup(@() fclose(id));

for k = 1:numel(lines)
    source = jsondecode(lines(k));
    messages = source.conversations;
    converted = repmat(struct('content', '', 'role', ''), numel(messages), 1);
    for m = 1:numel(messages)
        if strcmp(messages(m).from, 'human')
            role = 'user';
        elseif strcmp(messages(m).from, 'gpt')
            role = 'assistant';
        else
            error('Unsupported role "%s" in %s.', messages(m).from, sourceFile);
        end
        content = strrep(messages(m).value, '\n', newline);
        if strcmp(role, 'user')
            content = strtrim(strrep(content, '<image>', ''));
            content = ['<image>' content];
        end
        converted(m).content = content;
        converted(m).role = role;
    end
    assert(sum(contains({converted.content}, '<image>')) == 1, ...
        'Sample %s must contain exactly one <image> token.', source.id);
    record = struct;
    record.messages = converted;
    record.images = {[imageDir '/' source.id imageExtension]};
    fprintf(id, '%s\n', jsonencode(record));
end
count = numel(lines);
end

