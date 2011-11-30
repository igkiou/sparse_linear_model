function [train_labels_numeric test_labels_numeric label_legend] = extractCaltechLabels(train_labels, test_labels);

label_legend = {};
numWords = 0;
train_labels_numeric = zeros(size(train_labels, 1), 1);
for iter = 1:size(train_labels, 1),
	tempString = deblank(train_labels(iter, :));
	ind = find(strcmp(label_legend, tempString));
	if (isempty(ind)),
		numWords = numWords + 1;
		label_legend{numWords} = tempString;
		train_labels_numeric(iter) = numWords;
	else
		train_labels_numeric(iter) = ind;
	end;
end;

test_labels_numeric = zeros(size(test_labels, 1), 1);
for iter = 1:size(test_labels, 1),
	tempString = deblank(test_labels(iter, :));
	ind = find(strcmp(label_legend, tempString));
	test_labels_numeric(iter) = ind;
end;
